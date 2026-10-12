"""Cell-level parsing shared by every statement format: amounts, dates, times, text."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable
from typing import Any

Cell = Any  # str | int | float | date | datetime | None

_TIME_SUFFIX = re.compile(r"[ T,]+(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:\s?[AaPp]\.?[Mm]\.?)?)\s*$")

DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d,%m,%Y", "%d %m %Y",
    "%d/%m/%y", "%d-%m-%y", "%d.%m.%y", "%d,%m,%y",
    "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y",
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y", "%d %b %y", "%d-%b-%y", "%d/%b/%y", "%d %b, %Y",
    "%d %B %Y", "%d-%B-%Y", "%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%B %d %Y",
]  # fmt: skip
_DAY_FIRST = {f for f in DATE_FORMATS if f.startswith("%d")}
_MONTH_FIRST = {"%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y"}
EXCEL_EPOCH = dt.datetime(1899, 12, 30)


def text(cell: Cell) -> str:
    """Normalised display text of a cell (newlines from PDF/Excel wrapping collapsed)."""
    if cell is None:
        return ""
    if isinstance(cell, dt.datetime):
        return cell.isoformat(sep=" ")
    if isinstance(cell, dt.date):
        return cell.isoformat()
    if isinstance(cell, float) and cell.is_integer():
        return str(int(cell))
    return re.sub(r"\s+", " ", str(cell)).strip()


def is_blank(cell: Cell) -> bool:
    return text(cell) in ("", "-", "--", "—", "nan", "None")


def parse_money(raw: Cell) -> float | None:
    """'1,23,456.78' · '1,234.00 Cr' · '(500)' · '−250' · '250-' · '₹ 1,000' · 'Rs. 500/-' · 1234.5 → signed float.

    Strict: anything other than currency markers, separators, signs and a Cr/Dr marker around the
    number means it is not an amount ("AB12" is a reference, not 12).
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = text(raw)
    if not s or s in ("-", "--", "—"):
        return None
    body = s.upper().replace("−", "-").replace("–", "-")
    trailing = None
    m = re.search(r"(?<![A-Z])(CR|DR|C|D)\.?\s*$", body)
    if m:
        trailing = m.group(1)[0]
        body = body[: m.start()]
    body = re.sub(r"(INR|RS\.?|₹|\$|€|£)", "", body)
    body = body.replace(",", "").replace(" ", "").replace("\u00a0", "")
    body = re.sub(r"/-$", "", body)
    neg = False
    if body.startswith("(") and body.endswith(")"):
        body, neg = body[1:-1], True
    if body.endswith("-") and not body.startswith("-"):
        body, neg = body[:-1], True
    body = body.removeprefix("+")
    if not re.fullmatch(r"-?(\d+(\.\d+)?|\.\d+)", body):
        return None
    v = float(body)
    if neg or trailing == "D":
        v = -abs(v)
    elif trailing == "C":
        v = abs(v)
    return v


def has_explicit_sign(raw: Cell) -> bool:
    if isinstance(raw, (int, float)):
        return raw < 0
    s = text(raw).upper()
    return s.startswith(("-", "−", "(")) or bool(re.search(r"\b(CR|DR)\.?\s*$|-\s*$", s))


def split_time(raw: str) -> tuple[str, dt.time | None]:
    m = _TIME_SUFFIX.search(raw)
    if not m:
        return raw, None
    return raw[: m.start()].strip(), parse_time(m.group(1))


def parse_time(raw: Cell) -> dt.time | None:
    if isinstance(raw, dt.datetime):
        return raw.time() if raw.time() != dt.time(0) else None
    if isinstance(raw, dt.time):
        return raw
    s = text(raw).upper().replace(".", "")
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p", "%I:%M%p", "%I:%M:%S%p"):
        try:
            return dt.datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    return None


def excel_serial(value: float) -> dt.datetime | None:
    if 20000 <= value <= 80000:  # 1954 … 2119
        return EXCEL_EPOCH + dt.timedelta(days=float(value))
    return None


class DateParser:
    """Learns the date format of one column from its values, then parses consistently.

    Banks disagree on everything: 13/07/2026, 13-Jul-26, 2026-07-13, "13,07,2026" (ICICI's CSV),
    Excel serials, real datetime cells. Choosing one format per column (the one that parses the
    most values) avoids reading 05/07 as 5 July on one row and May 7 on the next.
    """

    def __init__(self, samples: Iterable[Cell]):
        strings = []
        for s in samples:
            if isinstance(s, (dt.date, dt.datetime)) or s is None:
                continue
            if isinstance(s, (int, float)):
                continue
            body, _ = split_time(text(s))
            if body:
                strings.append(body)
        self.fmt = self._choose(strings[:400])

    @staticmethod
    def _choose(samples: list[str]) -> str | None:
        if not samples:
            return None
        scores: dict[str, int] = {}
        for fmt in DATE_FORMATS:
            ok = 0
            for s in samples:
                try:
                    dt.datetime.strptime(s, fmt)
                    ok += 1
                except ValueError:
                    pass
            if ok:
                scores[fmt] = ok
        if not scores:
            return None
        best = max(scores.values())
        tied = [f for f in DATE_FORMATS if scores.get(f) == best]  # list order = preference (day-first)
        day_first = [f for f in tied if f in _DAY_FIRST]
        month_first = [f for f in tied if f in _MONTH_FIRST]
        if day_first and month_first:
            # both parse every sample (all days ≤ 12): India/UK default is day-first
            return day_first[0]
        return tied[0]

    def parse(self, cell: Cell) -> tuple[dt.date | None, dt.time | None]:
        if isinstance(cell, dt.datetime):
            return cell.date(), (cell.time() if cell.time() != dt.time(0) else None)
        if isinstance(cell, dt.date):
            return cell, None
        if isinstance(cell, (int, float)) and not isinstance(cell, bool):
            d = excel_serial(float(cell))
            return (d.date(), None) if d else (None, None)
        body, tm = split_time(text(cell))
        if not body:
            return None, None
        fmts = [self.fmt] if self.fmt else []
        fmts += [f for f in DATE_FORMATS if f != self.fmt]
        for fmt in fmts:
            try:
                return dt.datetime.strptime(body, fmt).date(), tm
            except ValueError:
                continue
        return None, None


def looks_like_date(cell: Cell) -> bool:
    if isinstance(cell, (dt.date, dt.datetime)):
        return True
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return False
    body, _ = split_time(text(cell))
    if not body or len(body) > 24 or not re.search(r"\d", body):
        return False
    return any(_try(body, f) for f in DATE_FORMATS)


def _try(s: str, fmt: str) -> bool:
    try:
        dt.datetime.strptime(s, fmt)
        return True
    except ValueError:
        return False
