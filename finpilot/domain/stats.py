"""Small, dependency-light statistics helpers."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Sequence

import numpy as np


def month_key(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_start(d: dt.date) -> dt.date:
    return d.replace(day=1)


def add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.month - 1 + n, 12)
    year, month = d.year + y, m + 1
    last = days_in_month(year, month)
    return dt.date(year, month, min(d.day, last))


def days_in_month(year: int, month: int) -> int:
    nxt = dt.date(year + (month == 12), (month % 12) + 1, 1)
    return (nxt - dt.date(year, month, 1)).days


def month_range(start: dt.date, end: dt.date) -> list[str]:
    out, cur = [], month_start(start)
    while cur <= end:
        out.append(month_key(cur))
        cur = add_months(cur, 1)
    return out


def robust_z(values: Sequence[float], x: float) -> float:
    """Median/MAD z-score — robust to the very outliers we are trying to find."""
    if len(values) < 3:
        return 0.0
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad if mad > 0 else (float(np.std(arr)) or 1.0)
    return (x - med) / scale


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    w = np.asarray(weights[-len(values) :], dtype=float)
    v = np.asarray(values, dtype=float)
    return float((v * w).sum() / w.sum()) if w.sum() else float(v.mean())


def pct_change(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new - old) / abs(old) * 100.0


def safe_round(x: float | None, nd: int = 2) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.percentile(arr, q)) if arr.size else 0.0
