"""Transaction ingestion: tolerant CSV parsing (real bank exports) and manual entry.

Handles: delimiter sniffing, BOMs, header synonyms (Narration/Particulars/Details…), single signed
amount *or* separate Debit/Credit columns *or* amount + DR/CR type column, Indian and US number
formats, dd/mm vs mm/dd detection, optional time/balance/category columns, and idempotent
re-imports (row fingerprints, so uploading the same statement twice adds nothing).
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from finpilot.decisions.jev import Evaluation, JevClient, get_jev
from finpilot.decisions.policy import log_decisions, record
from finpilot.decisions.spec import csv_column_questions, csv_column_state
from finpilot.domain.normalize import merchant_key
from finpilot.domain.taxonomy import CATEGORY_IDS
from finpilot.models import ImportBatch, Transaction, User

HEADERS = {
    "date": [
        "date",
        "txn date",
        "transaction date",
        "value date",
        "posting date",
        "posted date",
        "tran date",
        "booking date",
    ],
    "time": ["time", "txn time", "transaction time"],
    "description": [
        "description",
        "narration",
        "details",
        "particulars",
        "remarks",
        "transaction details",
        "memo",
        "payee",
        "merchant",
        "transaction remarks",
        "description/narration",
        "name",
    ],
    "amount": ["amount", "amt", "transaction amount", "amount (inr)", "amount(inr)", "value"],
    "debit": [
        "debit",
        "withdrawal",
        "withdrawal amt",
        "withdrawal amount",
        "dr",
        "debit amount",
        "paid out",
        "withdrawals",
        "money out",
    ],
    "credit": [
        "credit",
        "deposit",
        "deposit amt",
        "deposit amount",
        "cr",
        "credit amount",
        "paid in",
        "deposits",
        "money in",
    ],
    "type": ["type", "dr/cr", "cr/dr", "transaction type", "debit/credit", "txn type"],
    "balance": ["balance", "closing balance", "running balance", "available balance", "balance (inr)"],
    "category": ["category"],
    "account": ["account", "account name", "account number"],
}
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d %b %Y",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %B %Y",
    "%b %d, %Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
]


@dataclass
class ParsedRow:
    date: dt.date
    time: dt.time | None
    description: str
    amount: float
    balance: float | None = None
    category: str | None = None
    account: str | None = None


@dataclass
class ParseResult:
    rows: list[ParsedRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    total: int = 0


def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9/() ]", "", (h or "").strip().lower().replace("_", " ")).strip()


def _map_headers(headers: list[str]) -> dict[str, str]:
    normed = {_norm_header(h): h for h in headers}
    mapping: dict[str, str] = {}
    for field_, options in HEADERS.items():
        for opt in options:
            if opt in normed and normed[opt] not in mapping.values():
                mapping[field_] = normed[opt]
                break
    if "description" not in mapping:  # fall back to any header containing a keyword
        for n, orig in normed.items():
            if any(k in n for k in ("narrat", "descr", "particular", "detail", "remark")):
                mapping["description"] = orig
                break
    return mapping


def parse_money(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "--"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s_up = s.upper()
    trailing = None
    m = re.search(r"\b(CR|DR)\.?$", s_up)
    if m:
        trailing = m.group(1)
        s_up = s_up[: m.start()]
    s_clean = re.sub(r"[^\d.\-]", "", s_up.replace(",", ""))
    if s_clean in ("", "-", ".", "-."):
        return None
    try:
        v = float(s_clean)
    except ValueError:
        return None
    if neg:
        v = -abs(v)
    if trailing == "DR":
        v = -abs(v)
    elif trailing == "CR":
        v = abs(v)
    return v


def _detect_date_format(samples: list[str]) -> str | None:
    best, best_ok = None, 0
    for fmt in DATE_FORMATS:
        ok = 0
        for s in samples:
            try:
                dt.datetime.strptime(s.strip(), fmt)
                ok += 1
            except ValueError:
                pass
        if ok > best_ok:
            best, best_ok = fmt, ok
    # dd/mm vs mm/dd: if both parse equally, prefer the one consistent with values > 12
    if best in ("%d/%m/%Y", "%m/%d/%Y"):
        firsts = [int(s.split("/")[0]) for s in samples if s.count("/") == 2 and s.split("/")[0].isdigit()]
        seconds = [int(s.split("/")[1]) for s in samples if s.count("/") == 2 and s.split("/")[1].isdigit()]
        if seconds and max(seconds) > 12 and (not firsts or max(firsts) <= 12):
            best = "%m/%d/%Y"
        elif firsts and max(firsts) > 12:
            best = "%d/%m/%Y"
    return best


def _parse_time(raw: str | None) -> dt.time | None:
    if not raw:
        return None
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"):
        try:
            return dt.datetime.strptime(raw.strip(), fmt).time()
        except ValueError:
            continue
    return None


def _has_required(mapping: dict[str, str]) -> bool:
    return {"date", "description"} <= mapping.keys() and bool({"amount", "debit", "credit"} & mapping.keys())


def _jev_map_headers(
    headers: list[str],
    mapping: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    jev: JevClient | None,
    use_jev: bool,
) -> list[tuple[str, str, str]]:
    """System One reads unmapped headers + sample values; code checks the values fit the role."""
    client = jev or get_jev()
    unmapped = [h for h in headers if h and h not in mapping.values()]
    if not (use_jev and client.available()) or not unmapped or not rows:
        return []

    def samples(h: str) -> list[str]:
        vals = [str(r.get(h) or "").strip() for r in rows[:40]]
        return [v for v in vals if v][:4]

    jobs = [
        (csv_column_state(h, samples(h), [o for o in headers if o and o != h]), csv_column_questions())
        for h in unmapped
    ]
    evals = client.evaluate_many(jobs, agent="ingestion", purpose="csv_column")
    best: dict[str, tuple[float, str, Evaluation]] = {}
    for h, ev in zip(unmapped, evals, strict=True):
        if not isinstance(ev, Evaluation):
            continue
        a = ev["role"]
        if (
            a.choice in ("ignore", "reference")
            or a.choice in mapping
            or a.confidence < client.accept_confidence
        ):
            continue
        if not _values_fit(a.choice, [str(r.get(h) or "") for r in rows[:200]]):
            continue  # Jev's guess contradicts the data (e.g. "date" but nothing parses as a date)
        if a.choice not in best or a.confidence > best[a.choice][0]:
            best[a.choice] = (a.confidence, h, ev)
    decisions, out = [], []
    for role, (conf, h, ev) in best.items():
        out.append((h, role, f"Jev, confidence {conf:.2f}, values verified"))
        decisions.append(
            record(ev, "role", agent="ingestion", purpose="csv_column", subject=h, action="accepted")
        )
    log_decisions(decisions)
    return out


def _values_fit(role: str, values: list[str]) -> bool:
    vals = [v.strip() for v in values if v and v.strip()]
    if not vals:
        return False
    if role == "date":
        return _detect_date_format(vals) is not None
    if role in ("amount", "debit", "credit", "balance"):
        return sum(parse_money(v) is not None for v in vals) >= 0.8 * len(vals)
    if role == "time":
        return sum(_parse_time(v) is not None for v in vals) >= 0.8 * len(vals)
    if role == "description":
        return sum(bool(re.search(r"[A-Za-z]{3}", v)) for v in vals) >= 0.8 * len(vals)
    return True


def parse_csv(content: bytes | str, *, jev: JevClient | None = None, use_jev: bool = True) -> ParseResult:
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    text = text.strip()
    res = ParseResult()
    if not text:
        res.errors.append("File is empty")
        return res
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    lines = text.splitlines()
    # skip bank preamble lines until we find a header row with a date-like column
    start = 0
    for i, line in enumerate(lines[:30]):
        cells = [_norm_header(c) for c in next(csv.reader([line], dialect))]
        if any(c in HEADERS["date"] for c in cells) and len(cells) >= 3:
            start = i
            break
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])), dialect=dialect)
    headers = reader.fieldnames or []
    mapping = _map_headers(headers)
    raw_rows = list(reader)
    if not _has_required(mapping):
        # Unfamiliar export ("Txn Posted On", "Narrative", "Amt in Rs")? Ask System One what each
        # unmapped column holds, then verify its answer against the actual values in code.
        for header, role, why in _jev_map_headers(headers, mapping, raw_rows, jev=jev, use_jev=use_jev):
            mapping[role] = header
            res.warnings.append(f"Column '{header}' read as {role} ({why})")
    res.mapping = mapping
    if not _has_required(mapping):
        res.errors.append(
            f"Could not find required columns (date, description, amount or debit/credit). "
            f"Found headers: {headers}"
        )
        return res
    res.total = len(raw_rows)
    fmt = _detect_date_format([r.get(mapping["date"], "") or "" for r in raw_rows[:200]])
    if fmt is None:
        res.errors.append("Could not understand the date format")
        return res
    all_positive = (
        "amount" in mapping
        and "type" not in mapping
        and all((parse_money(r.get(mapping["amount"])) or 0) >= 0 for r in raw_rows)
    )
    if all_positive:
        res.warnings.append(
            "All amounts are positive and there is no debit/credit column — treating rows as "
            "spending except obvious credits (salary, refunds, interest)."
        )
    for n, r in enumerate(raw_rows, start=start + 2):
        try:
            raw_date = (r.get(mapping["date"]) or "").strip()
            d = dt.datetime.strptime(raw_date, fmt)
            desc = re.sub(r"\s+", " ", (r.get(mapping["description"]) or "").strip())
            if not desc:
                raise ValueError("empty description")
            if "amount" in mapping:
                amt = parse_money(r.get(mapping["amount"]))
                if amt is None:
                    raise ValueError("missing amount")
                if "type" in mapping:
                    t = (r.get(mapping["type"]) or "").strip().upper()
                    if t.startswith(("D", "W")):
                        amt = -abs(amt)
                    elif t.startswith(("C", "DEP")):
                        amt = abs(amt)
                elif all_positive:
                    credit_hint = re.search(
                        r"SALARY|REFUND|INTEREST|CASHBACK|NEFT CR|IMPS CR|UPI/CR|DIVIDEND", desc.upper()
                    )
                    amt = abs(amt) if credit_hint else -abs(amt)
            else:
                debit = parse_money(r.get(mapping.get("debit", ""))) or 0.0
                credit = parse_money(r.get(mapping.get("credit", ""))) or 0.0
                amt = abs(credit) - abs(debit)
            if amt == 0:
                raise ValueError("zero amount")
            cat = (r.get(mapping["category"]) or "").strip().lower() if "category" in mapping else None
            res.rows.append(
                ParsedRow(
                    date=d.date(),
                    time=_parse_time(r.get(mapping["time"]))
                    if "time" in mapping
                    else (d.time() if d.time() != dt.time(0, 0) else None),
                    description=desc[:500],
                    amount=round(float(amt), 2),
                    balance=parse_money(r.get(mapping["balance"])) if "balance" in mapping else None,
                    category=cat if cat in CATEGORY_IDS else None,
                    account=(r.get(mapping["account"]) or None) if "account" in mapping else None,
                )
            )
        except (ValueError, TypeError) as exc:
            if len(res.errors) < 25:
                res.errors.append(f"Row {n}: {exc}")
    return res


def fingerprint(row: ParsedRow, occurrence: int) -> str:
    key = f"{row.date.isoformat()}|{row.time or ''}|{row.description.upper()}|{row.amount:.2f}|{occurrence}"
    return hashlib.sha256(key.encode()).hexdigest()[:40]


def import_rows(
    session: Session,
    user: User,
    rows: list[ParsedRow],
    *,
    source: str,
    filename: str | None = None,
    errors: list[str] | None = None,
    total: int | None = None,
) -> ImportBatch:
    batch = ImportBatch(
        user_id=user.id,
        source=source,
        filename=filename,
        rows_total=total or len(rows),
        rows_invalid=(total or len(rows)) - len(rows),
        errors=(errors or [])[:25],
    )
    session.add(batch)
    session.flush()
    existing = set(session.scalars(select(Transaction.fingerprint).where(Transaction.user_id == user.id)))
    seen: Counter[tuple] = Counter()
    imported = dup = 0
    for r in rows:
        k = (r.date, r.time, r.description.upper(), round(r.amount, 2))
        seen[k] += 1
        fp = fingerprint(r, seen[k])
        if fp in existing:
            dup += 1
            continue
        existing.add(fp)
        merchant, _ = merchant_key(r.description)
        session.add(
            Transaction(
                user_id=user.id,
                batch_id=batch.id,
                date=r.date,
                time=r.time,
                description=r.description,
                merchant=merchant,
                amount=r.amount,
                account=r.account,
                fingerprint=fp,
                category=r.category,
                category_source="import" if r.category else None,
                category_confidence=1.0 if r.category else None,
                category_reason="Provided in the imported file" if r.category else None,
            )
        )
        imported += 1
    batch.rows_imported, batch.rows_duplicate = imported, dup
    _update_balance_anchor(user, rows)
    session.commit()
    return batch


def _update_balance_anchor(user: User, rows: list[ParsedRow]) -> None:
    with_bal = [r for r in rows if r.balance is not None]
    if with_bal:
        # statements may be newest-first; the row with the latest date (last in file order among ties
        # for chronological files) carries the closing balance
        chronological = rows[0].date <= rows[-1].date
        ordered = with_bal if chronological else list(reversed(with_bal))
        last = max(ordered, key=lambda r: r.date)
        candidates = [r for r in ordered if r.date == last.date]
        anchor = candidates[-1]
        if user.balance_as_of is None or anchor.date >= user.balance_as_of:
            user.current_balance, user.balance_as_of = float(anchor.balance or 0.0), anchor.date
    elif user.balance_as_of is None and rows:
        # no balance information: start from zero the day before the first transaction
        user.current_balance = user.current_balance or 0.0
        user.balance_as_of = min(r.date for r in rows) - dt.timedelta(days=1)
