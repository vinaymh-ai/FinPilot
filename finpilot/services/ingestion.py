"""Transaction ingestion: statement files from any bank, manual entry, idempotent imports.

Parsing lives in `finpilot.services.statements` (CSV/TSV/TXT, Excel .xlsx/.xls, HTML tables, PDF);
this module turns parsed rows into `Transaction`s: row fingerprints make re-importing the same
statement a no-op, and the statement's running balance anchors the user's current balance.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from finpilot.domain.normalize import merchant_key
from finpilot.domain.taxonomy import CATEGORY_IDS
from finpilot.models import ImportBatch, Transaction, User
from finpilot.services.statements import SUPPORTED_EXTENSIONS, StatementError, parse_statement
from finpilot.services.statements.values import parse_money

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ParseResult",
    "ParsedRow",
    "StatementError",
    "import_rows",
    "parse_csv",
    "parse_file",
    "parse_money",
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
    format: str = ""
    source: str = ""
    checks: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def parse_file(
    content: bytes,
    filename: str | None = None,
    *,
    password: str | None = None,
    jev: Any = None,
    use_jev: bool = True,
) -> ParseResult:
    """Parse a statement file. Raises `StatementError` when the file itself can't be read."""
    st = parse_statement(content, filename, password=password, jev=jev, use_jev=use_jev)
    rows = [
        ParsedRow(
            date=r.date,
            time=r.time,
            description=r.description,
            amount=r.amount,
            balance=r.balance,
            category=r.category if r.category in CATEGORY_IDS else None,
            account=r.account,
        )
        for r in st.rows
    ]
    return ParseResult(
        rows=rows,
        errors=st.errors,
        warnings=st.warnings,
        mapping=st.mapping,
        total=st.total,
        format=st.format,
        source=st.source,
        checks=st.checks,
        summary=st.summary(),
    )


def parse_csv(content: bytes | str, *, jev: Any = None, use_jev: bool = True) -> ParseResult:
    """Backwards-compatible entry point for CSV text; never raises (errors are in the result)."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    try:
        return parse_file(data, "statement.csv", jev=jev, use_jev=use_jev)
    except StatementError as exc:
        return ParseResult(errors=[str(exc)])


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
    if source != "demo" and rows and user.balance_source == "demo":
        user.current_balance, user.balance_as_of, user.balance_source = 0.0, None, "derived"
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
            user.balance_source = "statement"
    elif user.balance_as_of is None and rows:
        # no balance information: start from zero the day before the first transaction
        user.current_balance = user.current_balance or 0.0
        user.balance_as_of = min(r.date for r in rows) - dt.timedelta(days=1)
        user.balance_source = "derived"
