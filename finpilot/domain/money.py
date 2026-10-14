"""Money formatting shared by every agent that writes text.

INR amounts use Indian digit grouping (₹7,75,014.14), matching the web UI; other currencies use
Western grouping. Paise/cents are shown only when present, so exact figures survive verbatim and
the numeric grounding check (which reads both groupings) can verify them.
"""

from __future__ import annotations

SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}


def _group(integer: str, currency: str) -> str:
    if currency != "INR" or len(integer) <= 3:
        return f"{int(integer):,}"
    head, tail = integer[:-3], integer[-3:]
    pairs = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    if head:
        pairs.insert(0, head)
    return ",".join(pairs + [tail])


def format_money(amount: float, currency: str = "INR", digits: int | None = None) -> str:
    """₹1,15,000 · ₹449.16 · -₹28,000 · $1,200.50 (digits=None keeps cents only when present)."""
    value = abs(float(amount))
    if digits is None:
        digits = 2 if round(value, 2) != round(value) else 0
    text = f"{value:.{digits}f}"
    integer, _, frac = text.partition(".")
    body = _group(integer, currency) + (f".{frac}" if frac else "")
    sign = "-" if amount < 0 and round(value, digits) != 0 else ""
    return f"{sign}{SYMBOLS.get(currency, '')}{body}"
