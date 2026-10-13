"""Recognise what each statement column holds from its header text.

Every bank names the same columns differently — "Withdrawal Amount(INR)", "Debit", "Dr Amount",
"Withdrawal Amt.", "Paid out" are all the debit column. Headers are normalised (case, punctuation,
currency markers) and matched against role vocabularies, most specific role first.
"""

from __future__ import annotations

import re

from finpilot.services.statements.values import Cell, text

ROLES = (
    "date", "value_date", "time", "description", "debit", "credit", "amount", "type", "balance",
    "reference", "account", "category", "serial",
)  # fmt: skip
REQUIRED_AMOUNT = {"amount", "debit", "credit"}

_TYPE = {
    "type", "txn type", "transaction type", "dr cr", "cr dr", "debit credit", "credit debit",
    "dr cr indicator", "cr dr indicator", "d c", "c d", "dc", "crdr", "drcr",
}  # fmt: skip
_SERIAL = {
    "s no",
    "sno",
    "sr no",
    "srno",
    "sl no",
    "slno",
    "serial",
    "serial no",
    "serial number",
    "no",
    "sr",
    "s n",
}
_DESCRIPTION = (
    "narrat", "narat", "description", "particular", "remark", "detail", "memo", "payee",
    "merchant", "beneficiary", "transaction info", "transaction information",
)  # fmt: skip


def normalise(header: Cell) -> str:
    h = text(header).lower()
    h = re.sub(r"\((?:inr|rs\.?|₹|in rs\.?|in inr|amount in inr)\)", " ", h)
    h = re.sub(r"\b(inr|rs\.?|₹)\b", " ", h)
    h = h.replace("&", " and ").replace("/", " ").replace("_", " ")
    h = re.sub(r"[^a-z0-9 ]", " ", h)
    return re.sub(r"\s+", " ", h).strip()


def classify(header: Cell) -> str | None:
    n = normalise(header)
    if not n:
        return None
    tokens = set(n.split())
    if n in _SERIAL:
        return "serial"
    if n in _TYPE:
        return "type"
    if (
        "date" in tokens
        or "posted" in tokens
        or n.endswith(" dt")
        or n in ("dt", "txn dt", "value dt", "tran dt")
    ):
        return "value_date" if "value" in tokens else "date"
    if tokens & {"time"} and not tokens & {"amount", "amt"}:
        return "time"
    if "balance" in n or tokens & {"bal"}:
        return "balance"
    both_sides = (tokens >= {"dr", "cr"}) or (tokens >= {"debit", "credit"})
    if both_sides:  # "Dr/Cr Amount" is one signed amount; "Debit/Credit" alone is the type flag
        return "amount" if tokens & {"amount", "amt"} else "type"
    if (
        "withdraw" in n
        or tokens & {"debit", "debits", "dr", "outflow", "spent"}
        or "paid out" in n
        or "money out" in n
    ):
        return "debit"
    if (
        "deposit" in n
        or tokens & {"credit", "credits", "cr", "inflow", "received"}
        or "paid in" in n
        or "money in" in n
    ):
        return "credit"
    if tokens & {"amount", "amt", "amounts"} or n == "value":
        return "amount"
    if any(k in n for k in _DESCRIPTION) or n in (
        "name",
        "info",
        "transaction",
        "transactions",
        "txn details",
    ):
        return "description"
    if tokens & {"chq", "cheque", "check", "ref", "reference", "utr", "instrument", "rrn"} or (
        "id" in tokens and tokens & {"txn", "transaction", "tran"}
    ):
        return "reference"
    if "account" in n or n in ("a c", "ac no", "a c no"):
        return "account"
    if "category" in n:
        return "category"
    return None


def map_columns(header_row: list[Cell]) -> dict[str, int]:
    """Role → column index for one header row (first occurrence wins, except date prefers non-value)."""
    roles: dict[str, int] = {}
    for idx, cell in enumerate(header_row):
        role = classify(cell)
        if role and role not in roles:
            roles[role] = idx
    if "date" not in roles and "value_date" in roles:
        roles["date"] = roles["value_date"]
    return roles


def header_score(header_row: list[Cell]) -> int:
    roles = map_columns(header_row)
    useful = {"date", "description", "debit", "credit", "amount", "balance", "type", "reference", "serial"}
    score = len(useful & roles.keys())
    if "date" not in roles or not (REQUIRED_AMOUNT & roles.keys()):
        return 0
    return score
