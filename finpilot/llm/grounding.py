"""Numeric grounding check for LLM-written text.

The advisor lets an LLM *phrase* recommendations, but the numbers must come from our own
computations. This module extracts every figure the model wrote and verifies it against the
facts we supplied. Text that cites a number we never gave it is rejected and replaced with the
deterministic template — a cheap, effective hallucination guard.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

_NUM_RE = re.compile(
    r"(?<![\w.])[-+]?(?:₹|rs\.?|inr|\$|€|£)?\s?(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?\s?(k|l|lakh|lakhs|cr|crore|m|%)?(?![\w])",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([T ][\d:.+Z-]*)?$")
_ISO_TOKEN = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_SCALE = {"k": 1e3, "l": 1e5, "lakh": 1e5, "lakhs": 1e5, "cr": 1e7, "crore": 1e7, "m": 1e6}


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """Return (value, is_percent) pairs found in text."""
    out: list[tuple[float, bool]] = []
    for m in _NUM_RE.finditer(text or ""):
        whole, frac, suffix = m.group(1), m.group(2), (m.group(3) or "").lower()
        try:
            value = float(whole.replace(",", "") + ("." + frac if frac else ""))
        except ValueError:
            continue
        is_pct = suffix == "%"
        if suffix in _SCALE:
            value *= _SCALE[suffix]
        out.append((value, is_pct))
    return out


def _collect(obj: Any, acc: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, str):
        if _ISO_DATE.match(obj.strip()):
            return  # "2026-12-20" is a date, not three amounts (its parts would whitelist ₹2,000)
        acc.extend(v for v, _ in extract_numbers(_ISO_TOKEN.sub("", obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, acc)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect(v, acc)


def allowed_numbers(*facts: Any) -> list[float]:
    acc: list[float] = []
    for f in facts:
        _collect(f, acc)
    derived: list[float] = []
    for v in acc:
        a = abs(v)
        derived += [a, a * 12, a / 12, round(a), round(a, -2), round(a, -3)]
        if a <= 1:
            derived.append(a * 100)  # probability 0.42 → "42%" (not: day 20 → "₹2,000")
        elif a <= 100:
            derived.append(a / 100)
    return acc + derived


def _close(value: float, candidates: Iterable[float]) -> bool:
    v = abs(value)
    for c in candidates:
        tol = max(1.0, 0.03 * abs(c))  # rounding to the nearest 1k/lakh in prose is fine
        if abs(v - abs(c)) <= tol:
            return True
    return False


def _money_marked(text: str) -> list[bool]:
    """Per extracted number: does it carry a currency symbol or a thousands separator?"""
    marks = []
    for m in _NUM_RE.finditer(text or ""):
        token = m.group(0)
        marks.append(bool(re.search(r"₹|rs\.?|inr|\$|€|£", token, re.IGNORECASE)) or "," in m.group(1))
    return marks


def check_grounding(text: str, *facts: Any, strict: bool = False) -> tuple[bool, list[float]]:
    """Return (ok, ungrounded_numbers). Small integers (counts, months, days) and years are ignored."""
    allowed = allowed_numbers(*facts)
    if strict:
        raw: list[float] = []
        for fact in facts:
            _collect(fact, raw)
        allowed = raw + [abs(v) * 100 for v in raw if 0 <= abs(v) <= 1]
    bad: list[float] = []
    if strict:
        dates = set(_ISO_TOKEN.findall(json.dumps(facts, default=str)))
        unsupported_dates = set(_ISO_TOKEN.findall(text)) - dates
        if unsupported_dates:
            return False, [float(d[:4]) for d in sorted(unsupported_dates)]
        text = _ISO_TOKEN.sub("", text)
    for (value, is_pct), money in zip(extract_numbers(text), _money_marked(text), strict=True):
        if not money and not strict and not is_pct and value <= 31 and float(value).is_integer():
            continue  # day-of-month / counts / month numbers
        if 2000 <= value <= 2100 and float(value).is_integer() and not is_pct and not money:
            continue  # years ("₹2,000" is money, not a year)
        close = any(abs(abs(value) - abs(c)) <= 0.01 for c in allowed) if strict else _close(value, allowed)
        if not close:
            bad.append(value)
    return (not bad), bad
