"""Fallback for PDF statements without ruled tables: parse transaction lines from the page text.

A line that starts with a date and ends with money figures is a transaction:

    13/07/2026  UPI/RAKESH/9640591331@ibl/Payment   30.00   5,119.49
    13 Jul 2026 NEFT SALARY ACME TECH        94,571.00 Cr   94,690.49 Cr

The last figure is the running balance, the one before it the amount (its sign comes from a Cr/Dr
marker or, failing that, from the balance moving up or down). Lines without a date that follow a
transaction are wrapped remarks.
"""

from __future__ import annotations

import re

from finpilot.services.statements.tabular import Row, TableResult, _join, _resolve_signs, reconcile
from finpilot.services.statements.values import DateParser, parse_money

_DATE = r"\d{1,2}[-/. ,](?:\d{1,2}|[A-Za-z]{3,9})[-/. ,]\d{2,4}"
_LINE = re.compile(rf"^(?P<date>{_DATE})\s+(?:(?:{_DATE})\s+)?(?P<rest>.+)$")
_MONEY = re.compile(r"(?<![\w.])(-?\(?[\d,]*\d\.\d{2}\)?)(\s*(?:Cr|Dr|CR|DR)\b\.?)?")
_SKIP = re.compile(r"opening balance|closing balance|page \d+|statement of account|total", re.IGNORECASE)


def parse_lines(lines: list[str]) -> TableResult:
    res = TableResult()
    candidates = [(m, ln) for ln in lines if (m := _LINE.match(ln.strip()))]
    if len(candidates) < 2:
        res.errors.append("No transaction lines recognised in the PDF text.")
        return res
    dates = DateParser(m.group("date") for m, _ in candidates)
    rows: list[Row] = []
    for ln in lines:
        ln = ln.strip()
        m = _LINE.match(ln)
        if not m:
            if rows and ln and not _MONEY.search(ln) and not _SKIP.search(ln) and len(ln) < 120:
                rows[-1].description = _join(rows[-1].description, ln)
            continue
        if _SKIP.search(ln):
            continue
        d, tm = dates.parse(m.group("date"))
        figures = list(_MONEY.finditer(m.group("rest")))
        if d is None or not figures:
            continue
        res.total += 1
        desc_end = figures[0].start() if len(figures) >= 2 else figures[-1].start()
        desc = m.group("rest")[:desc_end].strip(" -|")
        if len(figures) >= 2:
            amt_m, bal_m = figures[-2], figures[-1]
            balance = parse_money(bal_m.group(0))
        else:
            amt_m, balance = figures[-1], None
        amount = parse_money(amt_m.group(0))
        if amount is None or amount == 0:
            continue
        marked = bool(amt_m.group(2)) or amt_m.group(1).startswith(("-", "("))
        rows.append(Row(d, tm, desc or "Transaction", round(amount, 2), balance, signed=marked))
    _resolve_signs(rows, res)
    res.checks = reconcile(rows, None)
    res.rows = rows
    res.mapping = {
        "date": "line start",
        "description": "line text",
        "amount": "second-last figure",
        "balance": "last figure",
    }
    res.warnings.append(
        "Read from the PDF's text layout (no table grid found) — please spot-check the amounts."
    )
    return res
