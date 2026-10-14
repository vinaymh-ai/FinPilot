"""Analyzer agent — studies behaviour over time and detects patterns & anomalies.

Everything here is computed from the user's own history (no generic tips):
* rolling 30-day windows vs the previous three 30-day windows → category spikes / drops
* robust (median/MAD) outliers per category, duplicate charges, fees, big first-time merchants
* recurring series → subscription inventory, overlaps, price hikes, *unused* memberships
  (actual unused status requires user confirmation, not absence of bank purchases)
* impulse purchases (late-night or outsized discretionary buys), weekday/weekend profile
An optional LLM step turns the computed facts into a short behaviour profile (grounded).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from finpilot.agents.base import BaseAgent
from finpilot.domain.recurring import RecurringCandidate, TxnLite, monthly_equivalent
from finpilot.domain.stats import add_months, month_key, pct_change, robust_z
from finpilot.domain.taxonomy import DISCRETIONARY, NON_SPEND, is_spending_flow, label
from finpilot.llm.grounding import check_grounding

WINDOW = 30
MEMBERSHIP_IDLE_DAYS = 45
IMPULSE_CATEGORIES = {"shopping", "food_delivery", "entertainment"}


@dataclass
class Finding:
    kind: str
    severity: str  # info | warning | critical | positive
    title: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)
    transaction_ids: list[int] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        key = self.metrics.get("key", self.title)
        return hashlib.sha1(f"{self.kind}:{key}".encode()).hexdigest()[:16]


@dataclass
class AnomalyFlag:
    txn_id: int
    kind: str
    reason: str
    score: float


@dataclass
class SubscriptionView:
    merchant: str
    category: str
    group: str | None
    cadence: str
    amount: float
    monthly_cost: float
    last_date: str
    next_expected: str
    status: str
    price_change_pct: float | None
    unused: bool
    usage_signal: str | None
    overlap_with: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    as_of: dt.date
    currency: str
    months: list[dict[str, Any]]
    windows: dict[str, Any]
    category_trends: list[dict[str, Any]]
    findings: list[Finding]
    anomalies: list[AnomalyFlag]
    subscriptions: list[SubscriptionView]
    impulse: dict[str, Any]
    behavior: dict[str, Any]
    profile: list[str] = field(default_factory=list)
    profile_model: str = "rules"
    data_quality: dict[str, Any] = field(default_factory=dict)

    @property
    def monthly_income(self) -> float:
        return float(self.windows.get("avg_income_30d", 0.0))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["findings"] = [dict(asdict(f), fingerprint=f.fingerprint) for f in self.findings]
        return d


SYSTEM_PROMPT = """You are FinPilot's Analyzer agent — a behavioural analyst for ONE user's bank ledger.
You receive computed facts: 30-day income and spending, category trends against the user's own previous windows,
top merchants, weekend share, impulse-style purchases, findings (spikes with the merchants driving them, anomalies,
subscriptions, month-to-date pace) and data-quality notes. Write exactly 3-4 observations, second person, each one
sentence, in this order:
1. the dominant spending pattern (largest discretionary category or merchant, with its amount);
2. the most important change versus the user's own history (only if a comparison window exists), naming the driver;
3. a risk worth attention (anomaly, obligation load, or review queue), or a positive trend if there is no risk;
4. optionally, a timing habit (weekend share or late-night purchases).
Rules: distinguish consumption from debt repayments, investments and transfers; a card bill reveals an obligation,
not what was bought. Amount and time alone do not prove impulse behaviour, and missing bank purchases do not prove a
membership is unused. If history is too short for a comparison, say so instead of inventing a trend. Do not
recommend actions or guess motives. Copy figures exactly as supplied (keep decimals) — never round or compute new
numbers. Currency: {currency}.
Respond with JSON only: {{"profile": ["...", "..."]}}"""


def _fmt(amount: float, currency: str) -> str:
    sym = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")
    digits = 2 if round(abs(amount), 2) != round(abs(amount)) else 0
    return f"{'-' if amount < 0 else ''}{sym}{abs(amount):,.{digits}f}"


class AnalyzerAgent(BaseAgent):
    name = "analyzer"
    role = "Aggregates behaviour over time; detects spikes, anomalies, subscriptions and impulse buys"
    system_prompt = SYSTEM_PROMPT

    def analyze(
        self,
        txns: list[TxnLite],
        recurring: list[RecurringCandidate],
        *,
        as_of: dt.date | None = None,
        currency: str = "INR",
        usage_overrides: dict[str, str] | None = None,
        narrate: bool = True,
        review_ids: set[int] | None = None,
    ) -> AnalysisReport:
        txns = [t for t in txns if as_of is None or t.date <= as_of]
        if not txns:
            return AnalysisReport(as_of or dt.date.today(), currency, [], {}, [], [], [], [], {}, {})
        as_of = as_of or max(t.date for t in txns)
        self.currency = currency
        recurring_ids = {i for r in recurring for i in r.txn_ids}
        findings: list[Finding] = []

        months = self._monthly(txns)
        windows = self._windows(txns, as_of)
        anomalies, anomaly_findings = self._anomalies(txns, as_of, recurring_ids)
        findings += anomaly_findings
        # one-off anomalies are reported on their own; they must not masquerade as a behaviour trend
        one_offs = {a.txn_id for a in anomalies if a.kind in ("large_purchase", "large_new_merchant")}
        self._one_offs = one_offs
        trends, trend_findings = self._category_trends(
            [t for t in txns if t.id not in one_offs], as_of, windows, recurring_ids
        )
        findings += trend_findings
        subs, sub_findings = self._subscriptions(txns, recurring, as_of, usage_overrides or {})
        findings += sub_findings
        impulse, impulse_findings = self._impulse(txns, as_of, recurring_ids)
        findings += impulse_findings
        behavior = self._behavior(txns, as_of)
        findings += self._cashflow_findings(windows, months)
        findings += self._pace(txns, as_of, recurring_ids)
        if review_ids:
            findings += self._review_queue(txns, review_ids, as_of)
        recent = [t for t in txns if as_of - dt.timedelta(days=30) < t.date <= as_of]
        cash_in = round(sum(t.amount for t in recent if t.amount > 0), 2)
        cash_out = round(sum(-t.amount for t in recent if t.amount < 0), 2)
        debt = [t for t in recent if t.amount < 0 and t.category in {"loans", "credit_card_payment"}]
        debt_total = round(sum(-t.amount for t in debt), 2)
        if debt and windows["last_30d_income"] > 0 and debt_total / windows["last_30d_income"] >= 0.4:
            findings.append(
                Finding(
                    "debt_obligations",
                    "warning",
                    "Repayments take a large share of recorded income",
                    f"Loan and credit-card repayments total {_fmt(debt_total, currency)} in the last 30 days. Card bills do not reveal the original purchases.",
                    {
                        "key": "debt",
                        "repayments": debt_total,
                        "income": windows["last_30d_income"],
                        "share_pct": round(debt_total / windows["last_30d_income"] * 100, 2),
                        "start": (as_of - dt.timedelta(days=29)).isoformat(),
                        "end": as_of.isoformat(),
                        "payments": [
                            {
                                "id": t.id,
                                "merchant": t.merchant,
                                "amount": -t.amount,
                                "date": t.date.isoformat(),
                                "category": t.category,
                            }
                            for t in debt
                        ],
                    },
                    [t.id for t in debt],
                )
            )
        windows["bank_cashflow"] = {
            "inflows": cash_in,
            "outflows": cash_out,
            "net": round(cash_in - cash_out, 2),
            "excluded_from_spending": round(
                cash_out - sum(-t.amount for t in recent if t.amount < 0 and t.category not in NON_SPEND), 2
            ),
            "transaction_ids": [t.id for t in recent],
        }
        if cash_in > 0 and cash_out >= cash_in * 0.95:
            findings.append(
                Finding(
                    "bank_cashflow_tight",
                    "warning",
                    "Bank outflows leave little recorded cash surplus",
                    f"Recorded cash inflows {_fmt(cash_in, currency)} and outflows {_fmt(cash_out, currency)} leave {_fmt(cash_in - cash_out, currency)} net; transfers and investments are included in bank cash flow.",
                    {
                        "key": "bank_net",
                        **windows["bank_cashflow"],
                        "start": (as_of - dt.timedelta(days=29)).isoformat(),
                        "end": as_of.isoformat(),
                    },
                    [t.id for t in recent],
                )
            )

        sev_rank = {"critical": 0, "warning": 1, "positive": 2, "info": 3}
        findings.sort(key=lambda f: (sev_rank.get(f.severity, 9), -abs(f.metrics.get("impact", 0))))
        report = AnalysisReport(
            as_of, currency, months, windows, trends, findings, anomalies, subs, impulse, behavior
        )
        coverage = (as_of - min(t.date for t in txns)).days + 1
        unknown = [t for t in txns if t.category in (None, "other")]
        report.data_quality = {
            "history_days": coverage,
            "uncategorized": len(unknown),
            "uncategorized_outflow": round(sum(-t.amount for t in unknown if t.amount < 0), 2),
            "comparison_available": coverage >= 60,
        }
        if narrate:
            report.profile, report.profile_model = self._profile(report)
        return report

    # ------------------------------------------------------------------ aggregates
    @staticmethod
    def _monthly(txns: list[TxnLite]) -> list[dict[str, Any]]:
        agg: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "income": 0.0,
                "spend": 0.0,
                "invested": 0.0,
                "by_category": defaultdict(float),
                "count": 0,
            }
        )
        for t in txns:
            m = agg[month_key(t.date)]
            m["count"] += 1
            cat = t.category or "other"
            if cat == "income":
                m["income"] += t.amount
            elif cat == "investments":
                m["invested"] += -t.amount
            elif is_spending_flow(cat, t.amount):
                m["spend"] += -t.amount
                m["by_category"][cat] += -t.amount
        out = []
        for k in sorted(agg):
            m = agg[k]
            savings = m["income"] - m["spend"]
            out.append(
                {
                    "month": k,
                    "income": round(m["income"], 2),
                    "spend": round(m["spend"], 2),
                    "invested": round(m["invested"], 2),
                    "savings": round(savings, 2),
                    "savings_rate": round(savings / m["income"] * 100, 1) if m["income"] > 0 else None,
                    "by_category": {
                        c: round(v, 2) for c, v in sorted(m["by_category"].items(), key=lambda kv: -kv[1])
                    },
                    "count": m["count"],
                }
            )
        return out

    @staticmethod
    def _window_sum(
        txns: list[TxnLite],
        start: dt.date,
        end: dt.date,
        cats: set[str] | None = None,
        exclude: set[str] = NON_SPEND,
    ) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for t in txns:
            if start < t.date <= end:
                c = t.category or "other"
                if c in exclude or not is_spending_flow(c, t.amount) or (cats and c not in cats):
                    continue
                out[c] += -t.amount
        return out

    def _windows(self, txns: list[TxnLite], as_of: dt.date) -> dict[str, Any]:
        def window(i: int) -> tuple[dt.date, dt.date]:
            end = as_of - dt.timedelta(days=WINDOW * i)
            return end - dt.timedelta(days=WINDOW), end

        first = min(t.date for t in txns)
        n_windows = max(1, min(4, ((as_of - first).days + 1) // WINDOW))
        res: dict[str, Any] = {"window_days": WINDOW, "available_windows": n_windows, "windows": []}
        for i in range(n_windows):
            s, e = window(i)
            inc = sum(t.amount for t in txns if s < t.date <= e and t.category == "income")
            spend = sum(v for v in self._window_sum(txns, s, e).values())
            res["windows"].append(
                {
                    "start": s.isoformat(),
                    "end": e.isoformat(),
                    "income": round(inc, 2),
                    "spend": round(spend, 2),
                }
            )
        incs = [w["income"] for w in res["windows"]]
        res["avg_income_30d"] = round(statistics.mean(incs), 2) if incs else 0.0
        res["last_30d_spend"] = res["windows"][0]["spend"]
        res["last_30d_income"] = res["windows"][0]["income"]
        prev = [w["spend"] for w in res["windows"][1:]]
        res["baseline_spend"] = round(statistics.mean(prev), 2) if prev else None
        return res

    def _category_trends(
        self, txns: list[TxnLite], as_of: dt.date, windows: dict[str, Any], recurring_ids: set[int]
    ) -> tuple[list[dict[str, Any]], list[Finding]]:
        n = windows["available_windows"]
        variable = [t for t in txns if t.id not in recurring_ids]
        per_window, per_window_var, counts = [], [], []
        for i in range(n):
            e = as_of - dt.timedelta(days=WINDOW * i)
            s = e - dt.timedelta(days=WINDOW)
            per_window.append(self._window_sum(txns, s, e))
            # behaviour signals use variable spend only: fixed bills landing twice in one 30-day
            # window (billing-date jitter) are not a behaviour change
            per_window_var.append(self._window_sum(variable, s, e))
            cnt: dict[str, int] = defaultdict(int)
            for t in variable:
                if s < t.date <= e:
                    cnt[t.category or "other"] += 1
            counts.append(cnt)
        income = max(windows.get("avg_income_30d") or 0.0, 1.0)
        cats = sorted({c for w in per_window for c in w})
        trends, findings = [], []
        for c in cats:
            recent = per_window[0].get(c, 0.0)
            hist = [w.get(c, 0.0) for w in per_window[1:]]
            baseline = statistics.mean(hist) if hist else None
            change = pct_change(recent, baseline) if baseline else None
            z = 0.0
            if len(hist) >= 2:
                sd = statistics.pstdev(hist) or max(baseline * 0.1, 1.0)
                z = (recent - baseline) / sd
            trend = {
                "category": c,
                "label": label(c),
                "last_30d": round(recent, 2),
                "baseline_30d": round(baseline, 2) if baseline is not None else None,
                "change_pct": round(change, 1) if change is not None else None,
                "z": round(z, 2),
                "history": [round(h, 2) for h in reversed(hist)],
            }
            trends.append(trend)
            if c in ("rent", "loans", "credit_card_payment", "insurance", "fees", "cash", "other"):
                continue
            recent = per_window_var[0].get(c, 0.0)
            hist = [w.get(c, 0.0) for w in per_window_var[1:]]
            baseline = statistics.mean(hist) if hist else None
            if not baseline or baseline <= 0:
                continue
            change = pct_change(recent, baseline)
            if len(hist) >= 2:
                sd = statistics.pstdev(hist) or max(baseline * 0.1, 1.0)
                z = (recent - baseline) / sd
            avg_count = statistics.mean([cn.get(c, 0) for cn in counts[1:]]) if len(counts) > 1 else 0
            if counts[0].get(c, 0) < 3 and avg_count < 3:
                continue  # lumpy categories (a single trip, a yearly fee) are not trends
            delta = recent - baseline
            material = max(1000.0, 0.015 * income)
            ids = [t.id for t in txns if t.category == c and t.date > as_of - dt.timedelta(days=WINDOW)]
            if change is not None and change >= 25 and delta >= material and (z >= 1.5 or len(hist) < 2):
                count_recent = len(ids)
                drivers = self._spike_drivers(variable, c, as_of, len(hist))
                driver_txt = (
                    " Biggest increases: "
                    + ", ".join(f"{d['merchant']} (+{_fmt(d['increase'], self.currency)})" for d in drivers)
                    + "."
                    if drivers
                    else ""
                )
                findings.append(
                    Finding(
                        "category_spike",
                        "warning" if change < 60 else "critical",
                        f"{label(c)} is up {change:.0f}% in the last 30 days",
                        f"You spent {_fmt(recent, self.currency)} on {label(c).lower()} in the last 30 days vs a "
                        f"{_fmt(baseline, self.currency)} average over the previous {len(hist)} months "
                        f"({count_recent} transactions).{driver_txt}",
                        {
                            "key": c,
                            "category": c,
                            "recent": round(recent, 2),
                            "baseline": round(baseline, 2),
                            "change_pct": round(change, 1),
                            "z": round(z, 2),
                            "impact": round(delta, 2),
                            "count": count_recent,
                            "drivers": drivers,
                        },
                        ids,
                    )
                )
            elif change is not None and change <= -25 and -delta >= material and c in DISCRETIONARY:
                findings.append(
                    Finding(
                        "category_drop",
                        "positive",
                        f"{label(c)} is down {abs(change):.0f}% — nice",
                        f"{_fmt(recent, self.currency)} in the last 30 days vs a {_fmt(baseline, self.currency)} average.",
                        {
                            "key": c,
                            "category": c,
                            "recent": round(recent, 2),
                            "baseline": round(baseline, 2),
                            "change_pct": round(change, 1),
                            "impact": round(delta, 2),
                        },
                    )
                )
        trends.sort(key=lambda t: -t["last_30d"])
        return trends, findings

    @staticmethod
    def _spike_drivers(
        variable: list[TxnLite], category: str, as_of: dt.date, n_hist: int
    ) -> list[dict[str, Any]]:
        """Merchants whose spend rose most in the latest window vs their own prior-window average."""
        if n_hist < 1:
            return []
        recent: dict[str, float] = defaultdict(float)
        prior: dict[str, float] = defaultdict(float)
        start = as_of - dt.timedelta(days=WINDOW)
        prior_start = start - dt.timedelta(days=WINDOW * n_hist)
        for t in variable:
            if t.category != category or not is_spending_flow(t.category, t.amount):
                continue
            if start < t.date <= as_of:
                recent[t.merchant] += -t.amount
            elif prior_start < t.date <= start:
                prior[t.merchant] += -t.amount
        rows = [
            {
                "merchant": m,
                "recent": round(v, 2),
                "baseline": round(prior.get(m, 0.0) / n_hist, 2),
                "increase": round(v - prior.get(m, 0.0) / n_hist, 2),
            }
            for m, v in recent.items()
        ]
        return sorted([r for r in rows if r["increase"] > 0], key=lambda r: -r["increase"])[:2]

    def _pace(self, txns: list[TxnLite], as_of: dt.date, recurring_ids: set[int]) -> list[Finding]:
        """Month-to-date variable spending vs the same days of last month (bills excluded)."""
        start = as_of.replace(day=1)
        days = (as_of - start).days + 1
        if days < 5:
            return []  # too early in the month for a meaningful pace
        prev_start = add_months(start, -1)
        prev_end = prev_start + dt.timedelta(days=days - 1)
        if prev_start < min(t.date for t in txns):
            return []

        def spend(a: dt.date, b: dt.date) -> tuple[float, list[int]]:
            rows = [
                t
                for t in txns
                if a <= t.date <= b
                and t.id not in recurring_ids
                and t.id not in self._one_offs
                and is_spending_flow(t.category, t.amount)
                and t.category not in ("rent", "loans", "credit_card_payment", "insurance")
            ]
            return round(sum(-t.amount for t in rows), 2), [t.id for t in rows]

        now, ids = spend(start, as_of)
        before, _ = spend(prev_start, prev_end)
        if before <= 0:
            return []
        change = round((now - before) / before * 100, 1)
        if abs(change) < 15 or abs(now - before) < 1000:
            return []
        return [
            Finding(
                "spending_pace",
                "warning" if change > 0 else "positive",
                f"Day-to-day spending is {'up' if change > 0 else 'down'} {abs(change):.0f}% vs the same point last month",
                f"{_fmt(now, self.currency)} in the first {days} days of {start:%B} vs {_fmt(before, self.currency)} "
                f"in the first {days} days of {prev_start:%B} (bills, EMIs and one-off anomalies excluded).",
                {
                    "key": "pace",
                    "month_to_date": now,
                    "same_point_last_month": before,
                    "change_pct": change,
                    "days": days,
                    "impact": round(max(0.0, now - before), 2),
                },
                ids,
            )
        ]

    def _review_queue(self, txns: list[TxnLite], review_ids: set[int], as_of: dt.date) -> list[Finding]:
        """Uncertain categories reduce every downstream number; surface them for confirmation."""
        rows = [t for t in txns if t.id in review_ids]
        if not rows:
            return []
        outflow = round(sum(-t.amount for t in rows if t.amount < 0), 2)
        total_out = sum(-t.amount for t in txns if t.amount < 0) or 1.0
        share = round(outflow / total_out * 100, 1)
        merchants = Counter(t.merchant for t in rows).most_common(5)
        return [
            Finding(
                "review_queue",
                "warning" if share >= 5 else "info",
                f"{len(rows)} transactions need a category check",
                f"{len(rows)} transactions ({_fmt(outflow, self.currency)} out, {share}% of recorded outflows) "
                f"have an uncertain category — mostly {', '.join(m for m, _ in merchants)}. Confirming them makes "
                "totals, trends and forecasts exact, and FinPilot remembers your answer for future imports.",
                {
                    "key": "review",
                    "count": len(rows),
                    "outflow": outflow,
                    "share_pct": share,
                    "merchants": [{"merchant": m, "count": n} for m, n in merchants],
                },
                [t.id for t in rows][:200],
            )
        ]

    # ------------------------------------------------------------------ anomalies
    def _anomalies(
        self, txns: list[TxnLite], as_of: dt.date, recurring_ids: set[int]
    ) -> tuple[list[AnomalyFlag], list[Finding]]:
        flags: dict[int, AnomalyFlag] = {}
        debits = [
            t for t in txns if t.amount < 0 and (t.category or "other") not in ("transfers", "investments")
        ]
        by_cat: dict[str, list[TxnLite]] = defaultdict(list)
        for t in debits:
            by_cat[t.category or "other"].append(t)
        all_amounts = [abs(t.amount) for t in debits]
        p95 = float(np.percentile(all_amounts, 95)) if all_amounts else 0.0
        recent_cut = as_of - dt.timedelta(days=90)

        # 1) outlier amount within its own category (robust z on log scale)
        for cat, items in by_cat.items():
            if cat in ("rent", "loans", "insurance", "cash"):
                continue
            logs = [np.log1p(abs(t.amount)) for t in items]
            med_amt = statistics.median(abs(t.amount) for t in items)
            for idx, (t, lg) in enumerate(zip(items, logs, strict=False)):
                if t.id in recurring_ids or t.date < recent_cut:
                    continue
                others = logs[:idx] + logs[idx + 1 :]
                z = robust_z(others, lg) if len(others) >= 5 else 0.0
                if z >= 3.5 and abs(t.amount) >= max(2000.0, 3 * med_amt):
                    flags[t.id] = AnomalyFlag(
                        t.id,
                        "large_purchase",
                        f"{_fmt(t.amount, self.currency)} is {abs(t.amount) / med_amt:.1f}× your "
                        f"typical {label(cat).lower()} spend ({_fmt(med_amt, self.currency)})",
                        round(z, 2),
                    )

        # 2) duplicate charges: same merchant & amount within 1 day, not a recurring series
        seen: dict[tuple[str, float], TxnLite] = {}
        for t in sorted(debits, key=lambda x: (x.date, x.time or dt.time())):
            k = (t.merchant, round(t.amount, 2))
            prev = seen.get(k)
            if (
                prev
                and (t.date - prev.date).days <= 1
                and abs(t.amount) >= 100
                and t.id not in recurring_ids
                and t.date >= recent_cut
            ):
                close_in_time = (
                    prev.time is None
                    or t.time is None
                    or (
                        abs(dt.datetime.combine(t.date, t.time) - dt.datetime.combine(prev.date, prev.time))
                        <= dt.timedelta(minutes=30)
                    )
                )
                if close_in_time:
                    flags[t.id] = AnomalyFlag(
                        t.id,
                        "duplicate_charge",
                        f"Same {_fmt(t.amount, self.currency)} charge from {t.merchant} twice within minutes",
                        3.0,
                    )
            seen[k] = t

        # 3) avoidable fees
        for t in by_cat.get("fees", []):
            if t.date >= recent_cut:
                flags.setdefault(
                    t.id, AnomalyFlag(t.id, "fee", f"Bank/late fee of {_fmt(t.amount, self.currency)}", 2.0)
                )

        # 4) first-time merchant with a large amount
        first_seen: dict[str, dt.date] = {}
        for t in sorted(txns, key=lambda x: x.date):
            first_seen.setdefault(t.merchant, t.date)
        for t in debits:
            if t.id in flags or t.id in recurring_ids or t.date < as_of - dt.timedelta(days=60):
                continue
            if (
                first_seen[t.merchant] == t.date
                and abs(t.amount) >= max(5000.0, p95)
                and t.category not in ("travel", "education", "health", "insurance", "rent")
            ):
                flags[t.id] = AnomalyFlag(
                    t.id,
                    "large_new_merchant",
                    f"First-ever payment to {t.merchant}: {_fmt(t.amount, self.currency)}",
                    2.5,
                )

        findings: list[Finding] = []
        by_id = {t.id: t for t in txns}
        for f in sorted(flags.values(), key=lambda f: -abs(by_id[f.txn_id].amount)):
            t = by_id[f.txn_id]
            sev = "warning" if f.kind in ("duplicate_charge", "fee", "large_purchase") else "info"
            title = {
                "large_purchase": f"Unusually large {label(t.category).lower()} purchase",
                "duplicate_charge": f"Possible duplicate charge at {t.merchant}",
                "fee": "You paid an avoidable fee",
                "large_new_merchant": f"Large first-time payment to {t.merchant}",
            }[f.kind]
            findings.append(
                Finding(
                    f"anomaly_{f.kind}",
                    sev,
                    title,
                    f"{f.reason} on {t.date:%d %b %Y}.",
                    {
                        "key": f"{f.kind}:{t.id}",
                        "amount": round(t.amount, 2),
                        "merchant": t.merchant,
                        "date": t.date.isoformat(),
                        "category": t.category,
                        "score": f.score,
                        "impact": abs(t.amount),
                    },
                    [t.id],
                )
            )
        return list(flags.values()), findings

    # ------------------------------------------------------------------ subscriptions
    def _subscriptions(
        self,
        txns: list[TxnLite],
        recurring: list[RecurringCandidate],
        as_of: dt.date,
        usage_overrides: dict[str, str],
    ) -> tuple[list[SubscriptionView], list[Finding]]:
        last_activity: dict[str, dt.date] = {}
        for t in txns:
            if t.amount < 0:
                last_activity[t.merchant] = max(last_activity.get(t.merchant, t.date), t.date)
        subs: list[SubscriptionView] = []
        for r in recurring:
            if r.kind != "subscription" or r.status != "active":
                continue
            unused, signal = False, None
            if r.parent:
                last = last_activity.get(r.parent)
                idle = (as_of - last).days if last else None
                if last is None or idle >= MEMBERSHIP_IDLE_DAYS:
                    signal = (
                        f"No {r.parent} orders since {last:%d %b} ({idle} days)"
                        if last
                        else f"No {r.parent} orders in your history"
                    ) + "; bank history cannot establish actual usage"
                else:
                    signal = f"Last {r.parent} order {idle} days ago"
            user_flag = usage_overrides.get(r.merchant)
            if user_flag == "rarely_used":
                unused, signal = True, "You marked this as rarely used"
            elif user_flag == "used":
                unused = False
            subs.append(
                SubscriptionView(
                    r.merchant,
                    r.category,
                    r.group,
                    r.cadence,
                    abs(r.typical_amount),
                    round(monthly_equivalent(abs(r.typical_amount), r.cadence), 2),
                    r.last_date.isoformat(),
                    r.next_expected.isoformat(),
                    r.status,
                    r.price_change_pct,
                    unused,
                    signal,
                )
            )

        groups: dict[str, list[SubscriptionView]] = defaultdict(list)
        for s in subs:
            if s.group:
                groups[s.group].append(s)
        findings: list[Finding] = []
        total = sum(s.monthly_cost for s in subs)
        if subs:
            n_unused = sum(1 for s in subs if s.unused)
            findings.append(
                Finding(
                    "subscription_inventory",
                    "info",
                    f"{len(subs)} active subscriptions costing {_fmt(total, self.currency)}/month",
                    f"{len(subs)} subscriptions ({', '.join(s.merchant for s in subs)})"
                    + (f"; {n_unused} look unused." if n_unused else "."),
                    {
                        "key": "inventory",
                        "count": len(subs),
                        "monthly_total": round(total, 2),
                        "unused": n_unused,
                        "items": [{"merchant": s.merchant, "monthly_cost": s.monthly_cost} for s in subs],
                    },
                )
            )
        for g, items in groups.items():
            if len(items) < 2:
                continue
            names = [s.merchant for s in items]
            for s in items:
                s.overlap_with = [n for n in names if n != s.merchant]
            cost = sum(s.monthly_cost for s in items)
            findings.append(
                Finding(
                    "subscription_overlap",
                    "warning",
                    f"{len(items)} overlapping {g.replace('_', ' ')} services",
                    f"You pay for {', '.join(names)} — {_fmt(cost, self.currency)}/month for the same kind of service.",
                    {
                        "key": g,
                        "group": g,
                        "monthly_total": round(cost, 2),
                        "items": [{"merchant": s.merchant, "monthly_cost": s.monthly_cost} for s in items],
                        "impact": round(cost - max(s.monthly_cost for s in items), 2),
                    },
                )
            )
        for s in subs:
            if s.unused:
                findings.append(
                    Finding(
                        "subscription_unused",
                        "warning",
                        f"{s.merchant} looks unused",
                        f"{s.usage_signal}, but you still pay {_fmt(s.monthly_cost, self.currency)}/month.",
                        {
                            "key": s.merchant,
                            "merchant": s.merchant,
                            "monthly_cost": s.monthly_cost,
                            "signal": s.usage_signal,
                            "impact": s.monthly_cost,
                        },
                    )
                )
            if s.price_change_pct and s.price_change_pct > 0:
                old = s.amount / (1 + s.price_change_pct / 100)
                findings.append(
                    Finding(
                        "subscription_price_hike",
                        "info",
                        f"{s.merchant} price went up {s.price_change_pct:.0f}%",
                        f"Now {_fmt(s.amount, self.currency)} per {s.cadence.rstrip('ly') or 'period'} "
                        f"(was about {_fmt(old, self.currency)}).",
                        {
                            "key": s.merchant,
                            "merchant": s.merchant,
                            "new_amount": s.amount,
                            "old_amount": round(old, 2),
                            "change_pct": s.price_change_pct,
                            "impact": round(s.amount - old, 2),
                        },
                    )
                )
        return subs, findings

    # ------------------------------------------------------------------ impulse & behaviour
    def _impulse(
        self, txns: list[TxnLite], as_of: dt.date, recurring_ids: set[int]
    ) -> tuple[dict[str, Any], list[Finding]]:
        disc = [
            t for t in txns if t.amount < 0 and t.category in IMPULSE_CATEGORIES and t.id not in recurring_ids
        ]
        medians = {
            c: statistics.median([abs(t.amount) for t in disc if t.category == c] or [0])
            for c in IMPULSE_CATEGORIES
        }
        tagged: dict[int, list[str]] = {}
        for t in disc:
            tags = []
            if t.time is not None and (t.time.hour >= 22 or t.time.hour < 4):
                tags.append("late_night")
            if t.category == "shopping" and medians["shopping"] and abs(t.amount) >= 3 * medians["shopping"]:
                tags.append("large_ticket")
            if tags:
                tagged[t.id] = tags
        start = as_of - dt.timedelta(days=WINDOW)
        recent = [t for t in disc if t.id in tagged and t.date > start]
        total = sum(-t.amount for t in recent)
        disc_total = sum(-t.amount for t in disc if t.date > start)
        last90 = [
            t
            for t in disc
            if t.id in tagged and t.date > as_of - dt.timedelta(days=90) and t.id not in self._one_offs
        ]
        info = {
            "tagged": tagged,
            "recent_count": len(recent),
            "recent_total": round(total, 2),
            "monthly_avg_90d": round(sum(-t.amount for t in last90) / 3, 2),
            "share_of_discretionary": round(total / disc_total * 100, 1) if disc_total else 0.0,
            "late_night_count": sum(1 for t in recent if "late_night" in tagged[t.id]),
        }
        findings = []
        if len(recent) >= 3 and total >= 2000:
            findings.append(
                Finding(
                    "impulse_spending",
                    "warning",
                    f"{_fmt(total, self.currency)} of impulse-style purchases in 30 days",
                    f"{len(recent)} late-night or outsized discretionary purchases — {info['share_of_discretionary']:.0f}% "
                    f"of your discretionary spend.",
                    {
                        "key": "impulse",
                        "count": len(recent),
                        "total": round(total, 2),
                        "share_pct": info["share_of_discretionary"],
                        "monthly_avg_90d": info["monthly_avg_90d"],
                        "impact": round(info["monthly_avg_90d"] / 2, 2),
                    },
                    [t.id for t in recent],
                )
            )
        return info, findings

    @staticmethod
    def _behavior(txns: list[TxnLite], as_of: dt.date) -> dict[str, Any]:
        start = as_of - dt.timedelta(days=90)
        disc = [t for t in txns if t.amount < 0 and t.category in DISCRETIONARY and t.date > start]
        total = sum(-t.amount for t in disc) or 1.0
        weekend = sum(-t.amount for t in disc if t.date.weekday() >= 5)
        merchants = Counter()
        for t in txns:
            if t.amount < 0 and t.date > as_of - dt.timedelta(days=WINDOW) and t.category not in NON_SPEND:
                merchants[t.merchant] += -t.amount
        weekday_spend = defaultdict(float)
        for t in disc:
            weekday_spend[t.date.strftime("%a")] += -t.amount
        return {
            "weekend_share_pct": round(weekend / total * 100, 1),
            "daily_discretionary_avg": round(total / 90, 2),
            "top_merchants_30d": [
                {"merchant": m, "amount": round(a, 2)} for m, a in merchants.most_common(6)
            ],
            "weekday_spend_90d": {k: round(v, 2) for k, v in weekday_spend.items()},
        }

    def _cashflow_findings(self, windows: dict[str, Any], months: list[dict[str, Any]]) -> list[Finding]:
        out: list[Finding] = []
        inc, spend = windows.get("last_30d_income", 0.0), windows.get("last_30d_spend", 0.0)
        if inc > 0:
            rate = (inc - spend) / inc * 100
            sev = "positive" if rate >= 20 else ("warning" if rate < 5 else "info")
            out.append(
                Finding(
                    "savings_rate",
                    sev,
                    f"Income minus spending is {rate:.0f}% before transfers and investments",
                    f"Income {_fmt(inc, self.currency)}, spending {_fmt(spend, self.currency)} "
                    f"(excluding transfers and investments); this is not the bank cash surplus.",
                    {
                        "key": "savings_rate",
                        "income": round(inc, 2),
                        "spend": round(spend, 2),
                        "savings_rate_pct": round(rate, 1),
                    },
                )
            )
        return out

    # ------------------------------------------------------------------ profile
    def _rules_profile(self, report: AnalysisReport) -> list[str]:
        """Deterministic behaviour profile: four specific observations computed from the user's data."""
        cur = report.currency
        out: list[str] = []
        disc = [t for t in report.category_trends if t["category"] in DISCRETIONARY and t["last_30d"] > 0]
        if disc:
            top = max(disc, key=lambda t: t["last_30d"])
            spend = report.windows.get("last_30d_spend") or 0
            share = f" — {top['last_30d'] / spend * 100:.0f}% of all spending" if spend else ""
            out.append(
                f"{top['label']} is your biggest discretionary category: {_fmt(top['last_30d'], cur)} in the "
                f"last 30 days{share}."
            )
        wk = report.behavior.get("weekday_spend_90d") or {}
        if wk:
            peak = max(wk.items(), key=lambda kv: kv[1])
            ws = report.behavior.get("weekend_share_pct", 0)
            habit = "weekend-heavy" if ws >= 40 else "spread across the week"
            out.append(
                f"Your discretionary spending is {habit} ({ws:.0f}% on Sat/Sun); {peak[0]} is your "
                f"heaviest day ({_fmt(peak[1], cur)} over 90 days)."
            )
        fixed = sum(s.monthly_cost for s in report.subscriptions)
        if report.impulse.get("late_night_count"):
            out.append(
                f"{report.impulse['late_night_count']} late-night purchases in the last 30 days — impulse-style "
                f"buys average {_fmt(report.impulse.get('monthly_avg_90d', 0), cur)}/month."
            )
        if fixed:
            inc = report.windows.get("avg_income_30d") or 0
            share = f" ({fixed / inc * 100:.1f}% of income)" if inc else ""
            out.append(
                f"{len(report.subscriptions)} subscriptions quietly cost {_fmt(fixed, cur)}/month{share}."
            )
        tops = report.behavior.get("top_merchants_30d") or []
        if len(out) < 4 and tops:
            out.append(
                f"Most money went to {tops[0]['merchant']} this month ({_fmt(tops[0]['amount'], cur)})."
            )
        return out[:4]

    # ------------------------------------------------------------------ LLM narrative
    def _profile(self, report: AnalysisReport) -> tuple[list[str], str]:
        facts = {
            "currency": report.currency,
            "last_30d_income": report.windows.get("last_30d_income"),
            "last_30d_spend": report.windows.get("last_30d_spend"),
            "top_categories_30d": [
                {k: t[k] for k in ("label", "last_30d", "baseline_30d", "change_pct")}
                for t in report.category_trends[:6]
            ],
            "weekend_share_pct": report.behavior.get("weekend_share_pct"),
            "top_merchants_30d": report.behavior.get("top_merchants_30d"),
            "impulse_30d": {
                k: report.impulse.get(k) for k in ("recent_count", "recent_total", "share_of_discretionary")
            },
            "findings": [{"title": f.title, "detail": f.detail} for f in report.findings[:8]],
            "data_quality": report.data_quality,
        }
        fallback = self._rules_profile(report)
        if not self.llm_available:
            return fallback, "rules"
        self.system_prompt = SYSTEM_PROMPT.format(currency=report.currency)
        data, label_, _err = self.safe_ask_json(facts, max_tokens=1400, temperature=0.1)
        bullets = data.get("profile") if isinstance(data, dict) else None
        if not isinstance(bullets, list) or not bullets:
            return fallback, "rules"
        good = [str(b) for b in bullets[:4] if check_grounding(str(b), facts, strict=True)[0]]
        verdicts = self.validate([(f"profile bullet {i + 1}", facts, b) for i, b in enumerate(good)])
        good = [b for b, v in zip(good, verdicts, strict=True) if v.ok]
        return (good, label_ or "llm") if good else (fallback, "rules")
