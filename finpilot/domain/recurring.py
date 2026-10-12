"""Recurring-payment (subscription / bill / salary) detection from raw transaction history.

Approach (no LLM needed):
1. group by merchant and direction (debit/credit);
2. split each group into amount clusters (a merchant like Amazon can have a fixed ₹299 Prime fee
   *and* random orders) — bills whose amount legitimately varies (electricity, salary) are also
   tried as a single cluster;
3. accept a cluster as a series when its inter-arrival times match a known cadence
   (weekly … yearly) with high regularity and roughly one charge per period.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from finpilot.domain.lexicon import rule_by_name
from finpilot.domain.stats import add_months, days_in_month


@dataclass
class TxnLite:
    id: int
    date: dt.date
    amount: float
    merchant: str
    category: str | None = None
    description: str = ""
    time: dt.time | None = None


@dataclass
class RecurringCandidate:
    merchant: str
    category: str
    kind: str
    cadence: str
    interval_days: float
    typical_amount: float
    last_amount: float
    first_date: dt.date
    last_date: dt.date
    next_expected: dt.date
    occurrences: int
    regularity: float
    price_change_pct: float | None
    status: str
    txn_ids: list[int] = field(default_factory=list)
    group: str | None = None
    parent: str | None = None


CADENCES: list[tuple[str, float, float]] = [
    ("weekly", 7.0, 1.5),
    ("biweekly", 14.0, 2.5),
    ("monthly", 30.44, 4.5),
    ("quarterly", 91.3, 10.0),
    ("yearly", 365.25, 20.0),
]
# categories whose amounts legitimately vary period to period
VARIABLE_AMOUNT_OK = {"utilities", "rent", "income", "loans", "insurance", "investments"}
SKIP_CATEGORIES = {"cash"}


def _amount_clusters(items: list[TxnLite], tol: float) -> list[list[TxnLite]]:
    ordered = sorted(items, key=lambda t: abs(t.amount))
    clusters: list[list[TxnLite]] = []
    for t in ordered:
        a = abs(t.amount)
        if clusters:
            ref = statistics.median(abs(x.amount) for x in clusters[-1])
            if ref and abs(a - ref) / ref <= tol:
                clusters[-1].append(t)
                continue
        clusters.append([t])
    return clusters


def _match_cadence(intervals: list[int]) -> tuple[str, float, float] | None:
    if not intervals:
        return None
    med = statistics.median(intervals)
    for name, nominal, tol in CADENCES:
        if abs(med - nominal) <= tol:
            within = sum(1 for i in intervals if abs(i - nominal) <= tol * 1.6)
            regularity = within / len(intervals)
            return name, nominal, regularity
    return None


SUBSCRIPTION_LIKE = {"subscriptions", "health", "entertainment", "education"}


def classify_kind(category: str, amount: float, group: str | None) -> str:
    if amount > 0:
        return "income"
    if category == "subscriptions" or (group and category in SUBSCRIPTION_LIKE):
        return "subscription"
    return {
        "rent": "rent",
        "utilities": "bill",
        "loans": "emi",
        "insurance": "insurance",
        "investments": "investment",
        "education": "bill",
    }.get(category, "other")


def _evaluate(
    cluster: list[TxnLite], as_of: dt.date, variable: bool, group_size: int | None = None
) -> RecurringCandidate | None:
    cluster = sorted(cluster, key=lambda t: t.date)
    # collapse same-day duplicates (they are anomalies, not cadence evidence)
    dedup: list[TxnLite] = []
    for t in cluster:
        if dedup and (t.date - dedup[-1].date).days <= 1:
            continue
        dedup.append(t)
    if len(dedup) < 2:
        return None
    intervals = [(b.date - a.date).days for a, b in zip(dedup, dedup[1:], strict=False)]
    match = _match_cadence(intervals)
    if match is None:
        return None
    cadence, nominal, regularity = match
    min_occ = 2 if cadence in ("quarterly", "yearly") else 3
    if len(dedup) < min_occ or regularity < 0.7:
        return None
    amounts = [abs(t.amount) for t in dedup]
    group_size = group_size or len(cluster)
    if len(dedup) == 2 and (group_size > 2 or abs(amounts[0] - amounts[1]) > 0.01 * max(amounts)):
        return None  # two look-alike charges inside a busy merchant are a coincidence, not a series
    if group_size >= 2 * len(cluster):
        # a fixed fee hidden inside a busy merchant (e.g. Amazon Prime among Amazon orders) has an
        # exact price; similar-but-different amounts there (cab rides, food orders) are coincidence
        med_amt = statistics.median(amounts)
        if len(dedup) < 4 or any(abs(a - med_amt) > 0.01 * med_amt for a in amounts):
            return None
    span = (dedup[-1].date - dedup[0].date).days
    expected = span / nominal + 1
    if len(dedup) > expected * 1.35 + 0.5:  # too many charges for this cadence → not a clean series
        return None
    if not variable:
        med = statistics.median(amounts)
        if med == 0 or max(abs(a - med) / med for a in amounts[-3:]) > 0.25:
            return None
    last = dedup[-1]
    prev = statistics.median(amounts[:-1]) if len(amounts) > 1 else amounts[-1]
    price_change = (amounts[-1] - prev) / prev * 100 if prev else None
    if price_change is not None and abs(price_change) < 2.0:
        price_change = None
    if cadence == "monthly":
        next_expected = next_monthly_slot([t.date for t in dedup])
    else:
        next_expected = last.date + dt.timedelta(days=round(nominal))
    lapsed = (as_of - last.date).days > nominal * 1.6 + 3
    category = last.category or "other"
    rule = rule_by_name(last.merchant)
    group = rule.group if rule else None
    parent = rule.parent if rule else None
    sign = 1 if last.amount > 0 else -1
    kind = classify_kind(category, last.amount, group)
    if kind not in ("subscription", "income"):
        price_change = None  # bills legitimately vary; only fixed-price plans have "price hikes"
    return RecurringCandidate(
        merchant=last.merchant,
        category=category,
        kind=kind,
        cadence=cadence,
        interval_days=round(statistics.mean(intervals), 1),
        typical_amount=round(sign * statistics.median(amounts[-3:]), 2),
        last_amount=round(last.amount, 2),
        first_date=dedup[0].date,
        last_date=last.date,
        next_expected=next_expected,
        occurrences=len(dedup),
        regularity=round(regularity, 2),
        price_change_pct=round(price_change, 1) if price_change is not None else None,
        status="lapsed" if lapsed else "active",
        txn_ids=[t.id for t in dedup],
        group=group,
        parent=parent,
    )


def next_monthly_slot(dates: list[dt.date]) -> dt.date:
    """Next due date of a monthly series, robust to early/late payments around month boundaries.

    Payroll "on the 1st" lands on the 27th–31st of the previous month when the 1st is a weekend. Such a
    payment belongs to the *next* month's slot, so the following payday is the 1st of the month after —
    naive `last + 1 month` would schedule it on the 27th and double-count it.
    """

    def early(d: dt.date) -> int:  # days late in a month count as "day ≤ 0" of the next month
        return d.day - days_in_month(d.year, d.month) if d.day >= 22 else d.day

    def late(d: dt.date) -> int:  # days early in a month count as "day > 28" of the previous month
        if d.day <= 7:
            prev = add_months(d.replace(day=1), -1)
            return days_in_month(prev.year, prev.month) + d.day
        return d.day

    mode = None
    if any(d.day <= 7 for d in dates) and any(d.day >= 22 for d in dates):
        sa, sb = statistics.pstdev([early(d) for d in dates]), statistics.pstdev([late(d) for d in dates])
        n_early = sum(d.day <= 7 for d in dates)
        mode = "early" if sa < sb or (sa == sb and n_early * 2 >= len(dates)) else "late"
    if mode == "early":
        natural = [d.day for d in dates if d.day < 22] or [early(d) for d in dates]
    elif mode == "late":
        natural = [d.day for d in dates if d.day > 7] or [late(d) for d in dates]
    else:
        natural = [d.day for d in dates]
    anchor = int(statistics.median(natural) + 0.5)
    last = dates[-1]
    slot = last.replace(day=1)
    if mode == "early" and last.day >= 22:
        slot = add_months(slot, 1)
    elif mode == "late" and last.day <= 7:
        slot = add_months(slot, -1)
    nxt = add_months(slot, 1)
    if anchor <= 0:
        return nxt - dt.timedelta(days=1 - anchor)
    return nxt.replace(day=min(anchor, days_in_month(nxt.year, nxt.month)))


def detect_recurring(txns: list[TxnLite], as_of: dt.date | None = None) -> list[RecurringCandidate]:
    if not txns:
        return []
    as_of = as_of or max(t.date for t in txns)
    groups: dict[tuple[str, bool], list[TxnLite]] = defaultdict(list)
    for t in txns:
        if t.category in SKIP_CATEGORIES or t.amount == 0:
            continue
        groups[(t.merchant, t.amount > 0)].append(t)

    found: list[RecurringCandidate] = []
    for (_merchant, _credit), items in groups.items():
        if len(items) < 2:
            continue
        cats = {t.category for t in items}
        # bills vary in amount; pure subscription merchants change price (hikes) — evaluate them whole
        if cats & VARIABLE_AMOUNT_OK or cats == {"subscriptions"}:
            whole = _evaluate(items, as_of, variable=True)
            if whole is not None:
                found.append(whole)
                continue
        for cluster in _amount_clusters(items, tol=0.08):
            cand = _evaluate(cluster, as_of, variable=False, group_size=len(items))
            if cand is not None:
                found.append(cand)
    found.sort(key=lambda c: c.typical_amount)
    return found


def monthly_equivalent(amount: float, cadence: str) -> float:
    factor = {"weekly": 52 / 12, "biweekly": 26 / 12, "monthly": 1.0, "quarterly": 1 / 3, "yearly": 1 / 12}
    return amount * factor.get(cadence, 1.0)
