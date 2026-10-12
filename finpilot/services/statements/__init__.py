"""Bank statement import for any bank and any common format.

    parse_statement(content, filename, password=None) -> StatementResult

CSV / TSV / TXT (any delimiter, any encoding), Excel (.xlsx and legacy .xls), HTML tables (the
".xls" many Indian banks actually serve), and PDF e-statements (ruled tables, with a text-layout
fallback; password-protected PDFs supported). Every format becomes grids of cells; one normaliser
(`tabular.parse_grid`) then finds the header, maps columns and reads transactions, so a new bank
layout needs no new code. Unfamiliar headers fall back to System One (Jev), whose answer is
verified against the column's actual values before it is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from finpilot.services.statements.columns import REQUIRED_AMOUNT, map_columns
from finpilot.services.statements.readers import (
    SUPPORTED_EXTENSIONS,
    Document,
    StatementError,
    Table,
    merge_tables,
    read,
)
from finpilot.services.statements.tabular import Row, TableResult, parse_grid
from finpilot.services.statements.textlines import parse_lines
from finpilot.services.statements.values import (
    DateParser,
    is_blank,
    looks_like_date,
    parse_money,
    parse_time,
    text,
)

log = logging.getLogger(__name__)

__all__ = ["SUPPORTED_EXTENSIONS", "StatementError", "StatementResult", "parse_statement"]


@dataclass
class StatementResult:
    rows: list[Row] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    total: int = 0
    format: str = ""
    source: str = ""
    checks: dict[str, Any] = field(default_factory=dict)
    opening_balance: float | None = None

    def summary(self) -> dict[str, Any]:
        dates = [r.date for r in self.rows]
        return {
            "format": self.format,
            "source": self.source,
            "rows": len(self.rows),
            "total": self.total,
            "mapping": self.mapping,
            "date_range": [min(dates).isoformat(), max(dates).isoformat()] if dates else None,
            "money_in": round(sum(r.amount for r in self.rows if r.amount > 0), 2),
            "money_out": round(-sum(r.amount for r in self.rows if r.amount < 0), 2),
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors[:10],
        }


def parse_statement(
    content: bytes,
    filename: str | None = None,
    *,
    password: str | None = None,
    jev: Any = None,
    use_jev: bool = True,
) -> StatementResult:
    if not content or not content.strip():
        raise StatementError("The file is empty.", code="empty")
    doc = read(content, filename, password)
    tables = merge_tables(doc.tables) if doc.format == "pdf" else doc.tables
    best: tuple[TableResult, Table] | None = None
    for table in tables:
        res = parse_grid(table.grid)
        if use_jev and (not res.rows or res.header_row in (None, -1)):
            # no recognised header: a verified Jev mapping of the real header row beats guessing by content
            assisted = _jev_assisted(table, res, jev=jev)
            if assisted is not None and assisted.rows and len(assisted.rows) >= len(res.rows):
                res = assisted
        if best is None or _better(res, best[0]):
            best = (res, table)
    if doc.format == "pdf" and doc.text_lines:
        # Text-strategy table detection can slice columns mid-word; compare it with reading the lines.
        res = parse_lines(doc.text_lines)
        if best is None or _better(res, best[0]):
            best = (res, Table([], "pdf:text"))
    out = StatementResult(format=doc.format)
    if best is None:
        out.errors.append("No table found in the file.")
        return out
    res, table = best
    out.rows, out.errors, out.warnings = res.rows, res.errors, res.warnings
    out.mapping, out.total, out.checks = res.mapping, res.total, res.checks
    out.opening_balance, out.source = res.opening_balance, table.source
    pct = res.checks.get("balance_reconciled_pct")
    if pct is not None and pct < 90:
        out.warnings.append(
            f"Only {pct:.0f}% of rows reconcile with the running balance — some amounts may be misread."
        )
    log.info(
        "statement parsed",
        extra={
            "event": "statement_parsed",
            "format": doc.format,
            "source": table.source,
            "rows": len(res.rows),
            "invalid": len(res.errors),
            "mapping": res.mapping,
            "reconciled_pct": pct,
        },
    )
    return out


def _quality(r: TableResult) -> tuple[float, float]:
    return (len(r.rows) - 2 * len(r.errors), r.checks.get("balance_reconciled_pct", 0.0))


def _better(a: TableResult, b: TableResult) -> bool:
    return _quality(a) > _quality(b)


# ----------------------------------------------------------------------------- Jev header fallback
def _jev_assisted(table: Table, failed: TableResult, *, jev: Any) -> TableResult | None:
    """Headers the vocabulary doesn't know ("When", "What", "Amt in Rs"): ask Jev, verify in code."""
    from finpilot.decisions.jev import Evaluation, get_jev
    from finpilot.decisions.policy import log_decisions, record
    from finpilot.decisions.spec import csv_column_questions, csv_column_state

    client = jev or get_jev()
    if not client.available():
        return None
    grid = table.grid
    first_data = next(
        (
            i
            for i, r in enumerate(grid[:200])
            if any(looks_like_date(c) for c in r)
            and any(parse_money(c) is not None and not looks_like_date(c) for c in r)
        ),
        None,
    )
    if not first_data:
        return None
    header_i = next(
        (i for i in range(first_data - 1, -1, -1) if sum(not is_blank(c) for c in grid[i]) >= 3), None
    )
    if header_i is None:
        return None
    header = grid[header_i]
    roles = map_columns(header)
    data = grid[header_i + 1 : header_i + 41]
    unmapped = [j for j, h in enumerate(header) if text(h) and j not in roles.values()]
    if not unmapped:
        return None
    others = [text(h) for h in header if text(h)]
    jobs = []
    for j in unmapped:
        samples = [text(r[j]) for r in data if j < len(r) and not is_blank(r[j])][:4]
        jobs.append(
            (
                csv_column_state(text(header[j]), samples, [o for o in others if o != text(header[j])]),
                csv_column_questions(),
            )
        )
    evals = client.evaluate_many(jobs, agent="ingestion", purpose="csv_column")
    best: dict[str, tuple[float, int, Evaluation]] = {}
    for j, ev in zip(unmapped, evals, strict=True):
        if not isinstance(ev, Evaluation):
            continue
        a = ev["role"]
        role = a.choice
        if role in ("ignore", "reference") or role in roles or a.confidence < client.accept_confidence:
            continue
        if not values_fit(role, [r[j] for r in grid[header_i + 1 :] if j < len(r)]):
            continue  # Jev's guess contradicts the data (e.g. "date" but nothing parses as a date)
        if role not in best or a.confidence > best[role][0]:
            best[role] = (a.confidence, j, ev)
    if not best:
        return None
    for role, (_, j, _) in best.items():
        roles[role] = j
    if "date" not in roles or not (REQUIRED_AMOUNT & roles.keys()):
        return None
    log_decisions(
        record(
            ev, "role", agent="ingestion", purpose="csv_column", subject=text(header[j]), action="accepted"
        )
        for (_, j, ev) in best.values()
    )
    res = parse_grid(grid, roles=roles, header_row=header_i)
    for role, (conf, j, _) in best.items():
        res.warnings.append(
            f"Column '{text(header[j])}' read as {role} (Jev, confidence {conf:.2f}, values verified)"
        )
    return res


def values_fit(role: str, values: list[Any]) -> bool:
    vals = [v for v in values if not is_blank(v)]
    if not vals:
        return False
    if role == "date":
        parser = DateParser(vals)
        return sum(parser.parse(v)[0] is not None for v in vals) >= 0.8 * len(vals)
    if role in ("amount", "debit", "credit", "balance"):
        return sum(parse_money(v) is not None for v in vals) >= 0.8 * len(vals)
    if role == "time":
        return sum(parse_time(v) is not None for v in vals) >= 0.8 * len(vals)
    if role == "description":
        return sum(any(ch.isalpha() for ch in text(v)) for v in vals) >= 0.8 * len(vals)
    return True


def describe_document(doc: Document) -> dict[str, Any]:  # pragma: no cover - debugging helper
    return {
        "format": doc.format,
        "tables": [(t.source, len(t.grid)) for t in doc.tables],
        "text_lines": len(doc.text_lines),
    }
