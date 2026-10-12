"""Grid → transactions: the bank-agnostic core of statement import.

Steps for one grid (a CSV, a sheet, a PDF table):
1. **Find the header row** anywhere in the first 80 rows (statements start with a preamble of
   account details) by scoring each row on recognised column roles.
2. **Map columns** to roles (date, description, debit/credit or amount, type, balance …).
   No header at all? Infer roles from the *contents* (a column of dates, numeric columns, the
   longest text column).
3. **Read rows**: one date format per column; multi-line remarks (rows with only text) are joined to
   the transaction above; footers, legends, repeated page headers and "opening balance" lines are
   skipped.
4. **Signs**: debit/credit columns → exact; a type column or Cr/Dr suffix → exact; otherwise the
   running **balance** decides (balance went down → money out); keywords are the last resort.
5. **Reconcile**: previous balance + amount = balance is checked for every row and reported.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any

from finpilot.services.statements.columns import REQUIRED_AMOUNT, header_score, map_columns
from finpilot.services.statements.values import (
    Cell,
    DateParser,
    has_explicit_sign,
    is_blank,
    looks_like_date,
    parse_money,
    parse_time,
    text,
)

HEADER_SCAN_ROWS = 80
_FOOTER = re.compile(
    r"^(legends?|total|totals|grand total|closing balance|opening balance|statement summary|page \d+|"
    r"generated on|this is a (computer|system) generated|end of statement|\*\*|note:|disclaimer|"
    r"carried forward|brought forward|b/f|c/f)",
    re.IGNORECASE,
)
_OPENING = re.compile(r"opening balance|balance b/?f|brought forward|b/f", re.IGNORECASE)
_CREDIT_HINT = re.compile(
    r"SALARY|REFUND|INTEREST|INT\.?PD|CASHBACK|NEFT CR|IMPS CR|UPI/CR|\bCR\b|DIVIDEND|BY TRANSFER|CREDIT|REVERSAL",
    re.IGNORECASE,
)


@dataclass
class Row:
    date: dt.date
    time: dt.time | None
    description: str
    amount: float
    balance: float | None = None
    category: str | None = None
    account: str | None = None
    signed: bool = True  # False while the sign is still unknown


@dataclass
class TableResult:
    rows: list[Row] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)  # role → header text
    header_row: int | None = None
    total: int = 0
    opening_balance: float | None = None
    checks: dict[str, Any] = field(default_factory=dict)


def find_header(grid: list[list[Cell]]) -> tuple[int | None, dict[str, int]]:
    best_i, best_score, best_map = None, 0, {}
    for i, row in enumerate(grid[:HEADER_SCAN_ROWS]):
        score = header_score(row)
        if score > best_score:
            best_i, best_score, best_map = i, score, map_columns(row)
    return best_i, best_map


def infer_by_content(grid: list[list[Cell]]) -> tuple[int | None, dict[str, int]]:
    """Headerless file: guess roles from what the columns contain."""
    data = [r for r in grid if any(looks_like_date(c) for c in r[:4])]
    if len(data) < 2:
        return None, {}
    width = max(len(r) for r in data)
    col = lambda j: [r[j] if j < len(r) else None for r in data]  # noqa: E731
    date_col = max(range(width), key=lambda j: sum(looks_like_date(c) for c in col(j)))
    numeric = []
    for j in range(width):
        if j == date_col:
            continue
        vals = [c for c in col(j) if not is_blank(c)]
        if vals and sum(parse_money(c) is not None for c in vals) >= 0.8 * len(vals):
            numeric.append(j)
    texts = [j for j in range(width) if j != date_col and j not in numeric]
    if not numeric or not texts:
        return None, {}
    desc_col = max(texts, key=lambda j: sum(len(text(c)) for c in col(j)))
    roles = {"date": date_col, "description": desc_col}
    if len(numeric) == 1:
        roles["amount"] = numeric[0]
    elif len(numeric) == 2:
        a, b = numeric
        sparse = lambda j: sum(is_blank(c) or parse_money(c) == 0 for c in col(j)) / len(data)  # noqa: E731
        if sparse(a) > 0.3 and sparse(b) > 0.3:  # each row fills one of them → debit / credit
            roles["debit"], roles["credit"] = a, b
        else:
            roles["amount"], roles["balance"] = a, b
    else:
        roles["debit"], roles["credit"], roles["balance"] = numeric[-3], numeric[-2], numeric[-1]
    return -1, roles


def parse_grid(
    grid: list[list[Cell]], roles: dict[str, int] | None = None, header_row: int | None = None
) -> TableResult:
    res = TableResult()
    grid = [list(r) for r in grid if r is not None]
    if roles is None:
        header_row, roles = find_header(grid)
        if header_row is None:
            header_row, roles = infer_by_content(grid)
            if header_row is not None:
                res.warnings.append("No header row found — columns were inferred from their contents.")
    if header_row is None or "date" not in roles or not (REQUIRED_AMOUNT & roles.keys()):
        res.errors.append("Could not find the date and amount columns.")
        return res
    if "description" not in roles:
        res.warnings.append("No description column found — using the reference column or a placeholder.")
    res.header_row = header_row
    header = grid[header_row] if header_row >= 0 else []
    res.mapping = {r: (text(header[i]) if i < len(header) else f"column {i + 1}") for r, i in roles.items()}

    body = grid[header_row + 1 :]
    get = lambda row, role: row[roles[role]] if role in roles and roles[role] < len(row) else None  # noqa: E731
    dates = DateParser(get(r, "date") for r in body)
    rows: list[Row] = []
    header_texts = {text(c).lower() for c in header if text(c)}
    started = False
    for n, raw in enumerate(body, start=header_row + 2):
        if not any(not is_blank(c) for c in raw):
            continue
        cells_lower = {text(c).lower() for c in raw if text(c)}
        if header_texts and len(cells_lower & header_texts) >= max(2, len(header_texts) // 2):
            continue  # header repeated on every PDF page
        desc = text(get(raw, "description")) or text(get(raw, "reference"))
        d, tm = dates.parse(get(raw, "date"))
        if d is None and "value_date" in roles and roles["value_date"] != roles["date"]:
            d, tm = dates.parse(get(raw, "value_date"))
        amount, signed = _amount(raw, roles, get)
        balance = parse_money(get(raw, "balance"))
        if d is None:
            if started and desc and amount is None and rows:
                rows[-1].description = _join(rows[-1].description, desc)  # wrapped remark line
                continue
            first = text(raw[0]) or desc
            if _OPENING.search(" ".join(text(c) for c in raw)) and balance is not None:
                res.opening_balance = balance
                continue
            if _FOOTER.search(first) or _FOOTER.search(desc):
                continue
            if amount is not None:
                res.total += 1
                _err(res, f"Row {n}: unreadable date {text(get(raw, 'date'))!r}")
            continue
        if amount is None:
            if _OPENING.search(desc) and balance is not None:
                res.opening_balance = balance
            elif balance is None and desc and started and rows and not _FOOTER.search(desc):
                rows[-1].description = _join(rows[-1].description, desc)
            continue
        res.total += 1
        if amount == 0:
            _err(res, f"Row {n}: zero amount")
            continue
        started = True
        if "time" in roles and tm is None:
            tm = parse_time(get(raw, "time"))
        cat = text(get(raw, "category")).lower() or None
        rows.append(
            Row(
                date=d,
                time=tm,
                description=(desc or "Transaction")[:500],
                amount=round(amount, 2),
                balance=balance,
                category=cat,
                account=text(get(raw, "account")) or None,
                signed=signed,
            )
        )
    if not ({"debit", "credit", "type"} & roles.keys()) and any(r.amount < 0 for r in rows):
        for r in rows:  # a single amount column that carries minus signs is signed throughout
            r.signed = True
    _resolve_signs(rows, res)
    res.checks = reconcile(rows, res.opening_balance)
    res.rows = rows
    return res


def _join(prev: str, more: str) -> str:
    # fixed-width wrapping splits tokens mid-word ("…@y" + "bl/Payment"): glue those back together
    if prev and more and prev[-1].isalnum() and more[0].isalnum() and " " not in prev[-12:]:
        return (prev + more)[:500]
    return f"{prev} {more}".strip()[:500]


def _err(res: TableResult, msg: str) -> None:
    if len(res.errors) < 25:
        res.errors.append(msg)


def _amount(raw: list[Cell], roles: dict[str, int], get) -> tuple[float | None, bool]:
    if "debit" in roles or "credit" in roles:
        debit = parse_money(get(raw, "debit"))
        credit = parse_money(get(raw, "credit"))
        if debit is None and credit is None:
            if "amount" in roles:
                return _single(raw, roles, get)
            return None, True
        return abs(credit or 0.0) - abs(debit or 0.0), True
    return _single(raw, roles, get)


def _single(raw: list[Cell], roles: dict[str, int], get) -> tuple[float | None, bool]:
    cell = get(raw, "amount")
    value = parse_money(cell)
    if value is None:
        return None, True
    if "type" in roles:
        t = text(get(raw, "type")).upper()
        if t.startswith(("D", "W", "-")):
            return -abs(value), True
        if t.startswith(("C", "DEP", "+")):
            return abs(value), True
    return value, has_explicit_sign(cell) or value < 0


def _resolve_signs(rows: list[Row], res: TableResult) -> None:
    unsigned = [r for r in rows if not r.signed]
    if not unsigned:
        return
    order = rows if (not rows or rows[0].date <= rows[-1].date) else list(reversed(rows))
    by_balance = 0
    prev_bal = res.opening_balance
    for r in order:
        if not r.signed and r.balance is not None and prev_bal is not None:
            delta = r.balance - prev_bal
            if abs(delta - abs(r.amount)) <= 0.011:
                r.amount, r.signed = abs(r.amount), True
                by_balance += 1
            elif abs(delta + abs(r.amount)) <= 0.011:
                r.amount, r.signed = -abs(r.amount), True
                by_balance += 1
        if r.balance is not None:
            prev_bal = r.balance
    # rows still unsigned (the first row without an opening balance, rows without a balance): keywords
    keyword = 0
    for r in rows:
        if not r.signed:
            r.amount = abs(r.amount) if _CREDIT_HINT.search(r.description) else -abs(r.amount)
            r.signed = True
            keyword += 1
    if by_balance:
        res.warnings.append(f"{by_balance} amount sign(s) inferred from the running balance.")
    if keyword:
        res.warnings.append(
            f"{keyword} amount sign(s) guessed from the description (salary/refund/interest → money in, "
            "everything else → money out). Check them on the Transactions page."
        )


def reconcile(rows: list[Row], opening: float | None) -> dict[str, Any]:
    """How many rows satisfy previous balance + amount = balance (a strong read-correctness signal)."""
    with_bal = [r for r in rows if r.balance is not None]
    if len(with_bal) < 2:
        return {}
    order = rows if rows[0].date <= rows[-1].date else list(reversed(rows))
    ok = checked = 0
    prev = opening
    for r in order:
        if r.balance is None:
            continue
        if prev is not None:
            checked += 1
            ok += abs(prev + r.amount - r.balance) <= 0.011
        prev = r.balance
    if not checked:
        return {}
    return {
        "balance_checked": checked,
        "balance_reconciled": ok,
        "balance_reconciled_pct": round(ok / checked * 100, 1),
    }
