"""Forecaster agent — projects the user's cash trajectory and simulates "what-if" decisions.

Model ("if you keep behaving like this"):
* **Known commitments** — every active recurring series (salary, rent, EMIs, bills, subscriptions)
  is scheduled on its own cadence and day-of-month.
* **Everything else** — daily variable cash-flow is bootstrapped from the user's own last 90 days,
  stratified by weekday (weekends look like weekends) and recency-weighted (the last 30 days count
  double), with one-off anomalies removed. 400 Monte-Carlo paths give P10/P50/P90 bands.
* **Goals** — funded in priority order from balance above a safety buffer; per-path funding
  gives a probability of hitting each goal, the projected shortfall and an ETA. A *spend* goal
  (a trip, a laptop — classified through API reasoning from its name) leaves the account on its deadline,
  so the balance bands, low point and overdraft risk include that outflow; a *reserve* goal
  (an emergency fund) stays in the account as earmarked savings.
* **What-if** — a scenario (one-off purchase, new recurring cost, category cut, cancelled
  subscriptions, income change) is replayed on the *same random draws* as the baseline
  (common random numbers), so the reported delta is the effect of the decision, not noise.
"""

from __future__ import annotations

import datetime as dt
import math
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from finpilot.agents.base import BaseAgent
from finpilot.domain.money import format_money
from finpilot.domain.recurring import RecurringCandidate, TxnLite
from finpilot.domain.stats import add_months, month_key
from finpilot.domain.taxonomy import DISCRETIONARY, ESSENTIAL, label
from finpilot.llm.grounding import check_grounding

LOOKBACK_DAYS = 90
RECENT_DAYS = 30
RECENT_WEIGHT = 2.0
MAX_HORIZON = 730


@dataclass
class GoalInput:
    id: int | None
    name: str
    target: float
    saved: float
    deadline: dt.date
    priority: int = 1
    spend_at_deadline: bool = False  # True → the funded amount leaves the account on the deadline


@dataclass
class Adjustment:
    """A what-if change. `amount` is positive for money going OUT unless `kind == "income"`."""

    type: str  # one_time | recurring | category_change | cancel_recurring | income_change
    amount: float = 0.0
    kind: str = "expense"  # expense | income
    date: dt.date | None = None
    end_date: dt.date | None = None
    cadence: str = "monthly"
    category: str | None = None
    pct: float | None = None
    merchants: list[str] = field(default_factory=list)
    label: str = ""

    @property
    def signed(self) -> float:
        return abs(self.amount) if self.kind == "income" else -abs(self.amount)

    def describe(self, currency: str = "INR") -> str:
        when = f" on {self.date:%d %b %Y}" if self.date else ""
        if self.type == "one_time":
            return f"{self.label or 'One-off ' + self.kind} of {format_money(abs(self.amount), currency, 0)}{when}"
        if self.type == "recurring":
            return f"{self.label or 'Recurring ' + self.kind} of {format_money(abs(self.amount), currency, 0)}/{self.cadence}"
        if self.type == "category_change":
            return f"{label(self.category)} spending {self.pct:+.0f}%"
        if self.type == "cancel_recurring":
            return f"Cancel {', '.join(self.merchants)}"
        if self.type == "income_change":
            return f"Income {self.pct:+.0f}%"
        return self.type


@dataclass
class _RunOutput:
    paths: np.ndarray  # (P, H) end-of-day balances
    expected: np.ndarray  # (H,)
    recurring_flow: np.ndarray  # (H,)


class ForecastEngine:
    def __init__(
        self,
        txns: list[TxnLite],
        recurring: list[RecurringCandidate],
        *,
        balance: float,
        as_of: dt.date,
        goals: list[GoalInput] | None = None,
        exclude_ids: set[int] | None = None,
        horizon_days: int = 180,
        n_paths: int = 400,
        seed: int = 42,
        safety_buffer: float | None = None,
        balance_known: bool = True,
    ):
        if not math.isfinite(balance) or not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError("Forecast requires a finite balance and at least one simulation path")
        if safety_buffer is not None and (not math.isfinite(safety_buffer) or safety_buffer < 0):
            raise ValueError("Safety buffer must be finite and nonnegative")
        if any(not math.isfinite(t.amount) for t in txns):
            raise ValueError("Transaction amounts must be finite")
        txns = [t for t in txns if t.date <= as_of]
        self.as_of = as_of
        self.balance_known = balance_known
        self.balance = float(balance) if balance_known else 0.0
        self.goals = sorted(goals or [], key=lambda g: (g.priority, g.deadline)) if balance_known else []
        self.recurring = [r for r in recurring if r.status == "active"]
        self.n_paths = n_paths
        self.seed = seed
        need = max([horizon_days] + [(g.deadline - as_of).days + 15 for g in self.goals])
        self.H = int(min(MAX_HORIZON, max(30, need)))
        self.dates = [as_of + dt.timedelta(days=i + 1) for i in range(self.H)]
        self._weekday = np.array([d.weekday() for d in self.dates])

        recurring_ids = {i for r in recurring for i in r.txn_ids}
        excluded = recurring_ids | (exclude_ids or set())
        first = min((t.date for t in txns), default=as_of)
        self.L = int(max(1, min(LOOKBACK_DAYS, (as_of - first).days + 1)))
        lb_dates = [as_of - dt.timedelta(days=self.L - 1 - i) for i in range(self.L)]
        idx_of = {d: i for i, d in enumerate(lb_dates)}
        cats: dict[str, int] = {}
        rows: list[tuple[int, int, float]] = []
        for t in txns:
            if t.id in excluded or t.date not in idx_of:
                continue
            ci = cats.setdefault(t.category or "other", len(cats))
            rows.append((ci, idx_of[t.date], t.amount))
        self.categories = list(cats)
        self.V = np.zeros((max(1, len(cats)), self.L))
        for ci, li, amt in rows:
            self.V[ci, li] += amt
        self.T = self.V.sum(axis=0)
        lb_wd = np.array([d.weekday() for d in lb_dates])
        recency = np.array([RECENT_WEIGHT if (as_of - d).days < RECENT_DAYS else 1.0 for d in lb_dates])
        self._pool: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for w in range(7):
            ids = np.where(lb_wd == w)[0]
            if ids.size == 0:
                ids = np.arange(self.L)
            p = recency[ids] / recency[ids].sum()
            self._pool[w] = (ids, p)
        self._idx = self._sample_indices()
        # One-off shocks (the anomalies we kept out of the bootstrap) still happen: model their observed
        # frequency and size as a compound-Poisson risk term so the P10 band reflects them.
        history_days = max(30, (as_of - first).days + 1)
        self.shock_pool = [
            t.amount
            for t in txns
            if t.id in (exclude_ids or set()) and t.id not in recurring_ids and t.amount < 0
        ]
        self.shock_rate = len(self.shock_pool) / history_days
        self._shocks = self._sample_shocks()
        self._recurring_amounts: dict[str, list[float]] = {}
        for r in self.recurring:
            if r.category not in {"credit_card_payment", "utilities", "income"}:
                continue
            observations = [t for t in txns if t.id in set(r.txn_ids)]
            if r.merchant == "Monthly card repayments":
                monthly: dict[str, float] = defaultdict(float)
                for t in observations:
                    monthly[month_key(t.date)] += t.amount
                amounts = list(monthly.values())
            else:
                amounts = [t.amount for t in observations]
            if len(amounts) >= 3 and np.ptp(amounts) > 0.01:
                self._recurring_amounts[r.merchant] = amounts
        self.safety_buffer = float(safety_buffer) if safety_buffer is not None else self._default_buffer()

    # ------------------------------------------------------------------ setup helpers
    def _sample_indices(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        idx = np.empty((self.n_paths, self.H), dtype=np.int32)
        for w in range(7):
            cols = np.where(self._weekday == w)[0]
            if cols.size == 0:
                continue
            ids, p = self._pool[w]
            # Each path also samples uncertainty in the observed weekday distribution.
            # Independent daily resampling alone treats a short history as a known spending rate
            # and understates multi-month uncertainty, especially for irregular income.
            weights = rng.dirichlet(p * len(ids), size=self.n_paths)
            for path in range(self.n_paths):
                idx[path, cols] = rng.choice(ids, size=cols.size, p=weights[path])
        return idx

    def _sample_shocks(self) -> np.ndarray:
        shocks = np.zeros((self.n_paths, self.H))
        if not self.shock_pool or self.shock_rate <= 0:
            return shocks
        rng = np.random.default_rng(self.seed + 1)
        counts = rng.poisson(self.shock_rate * self.H, size=self.n_paths)
        pool = np.asarray(self.shock_pool)
        for p, n in enumerate(counts):
            if n:
                np.add.at(shocks[p], rng.integers(0, self.H, size=n), rng.choice(pool, size=n))
        return shocks

    def _weekday_mean(self, flows: np.ndarray) -> np.ndarray:
        """Recency-weighted expected flow per horizon day (deterministic companion of the bootstrap)."""
        means = np.zeros(7)
        for w in range(7):
            ids, p = self._pool[w]
            means[w] = float((flows[ids] * p).sum())
        return means[self._weekday]

    def _default_buffer(self) -> float:
        monthly_fixed = sum(
            abs(r.typical_amount) * _per_month(r.cadence)
            for r in self.recurring
            if r.typical_amount < 0 and r.kind != "investment"
        )
        ess = sum(-self.V[i].sum() for i, c in enumerate(self.categories) if c in ESSENTIAL)
        ess_monthly = max(0.0, ess) / self.L * 30.4
        buf = 0.5 * (monthly_fixed + ess_monthly)
        return float(max(5000.0, round(buf, -3)))

    def _schedule(
        self, cancelled: set[str], income_mult: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        flow = np.zeros(self.H)
        events: list[dict[str, Any]] = []
        for r in self.recurring:
            if r.merchant in cancelled:
                continue
            d = r.next_expected
            overdue = (self.as_of - d).days
            if overdue >= 0:
                if overdue <= max(3, r.interval_days / 3):
                    d = self.as_of + dt.timedelta(days=1)
                else:
                    d = _advance(d, r)
            n = 0
            while d <= self.dates[-1] and n < 1000:
                i = (d - self.as_of).days - 1
                if i >= 0:
                    amt = r.typical_amount * (income_mult[i] if r.typical_amount > 0 else 1.0)
                    flow[i] += amt
                    events.append(
                        {
                            "date": d.isoformat(),
                            "merchant": r.merchant,
                            "amount": round(amt, 2),
                            "kind": r.kind,
                            "category": r.category,
                        }
                    )
                d = _advance(d, r)
                n += 1
        events.sort(key=lambda e: e["date"])
        return flow, events

    # ------------------------------------------------------------------ core run
    def run(self, adjustments: list[Adjustment] | None = None) -> tuple[_RunOutput, list[dict[str, Any]]]:
        adjustments = adjustments or []
        cancelled = {m for a in adjustments if a.type == "cancel_recurring" for m in a.merchants}
        income_mult = np.ones(self.H)
        cat_mult: dict[str, np.ndarray] = {}
        extra = np.zeros(self.H)
        for a in adjustments:
            if a.type not in {
                "one_time",
                "recurring",
                "category_change",
                "cancel_recurring",
                "income_change",
            }:
                raise ValueError("Unknown scenario adjustment type")
            if (
                not math.isfinite(a.amount)
                or a.amount < 0
                or (a.pct is not None and (not math.isfinite(a.pct) or a.pct < -100))
            ):
                raise ValueError(
                    "Scenario amounts must be finite and nonnegative; percentage cuts cannot exceed 100%"
                )
            if a.date and (
                a.date < self.as_of
                or a.date > self.dates[-1]
                or (a.type == "one_time" and a.date == self.as_of)
            ):
                raise ValueError("Scenario date must be within the future forecast horizon")
            if a.end_date and (a.end_date < (a.date or self.dates[0]) or a.end_date > self.dates[-1]):
                raise ValueError("Scenario end date must follow its start within the forecast horizon")
            if a.type == "one_time" and a.date is None:
                raise ValueError("One-time scenarios require a date")
            if a.type == "category_change" and a.category not in self.categories:
                raise ValueError("Category has no variable cash flow in this forecast")
            start_i = max(0, self._index(a.date)) if a.date else 0
            end_i = self._index(a.end_date) + 1 if a.end_date else self.H
            if a.type == "income_change" and a.pct is not None:
                income_mult[start_i:end_i] *= 1 + a.pct / 100
            elif a.type == "category_change" and a.category and a.pct is not None:
                m = cat_mult.setdefault(a.category, np.ones(self.H))
                m[start_i:end_i] *= 1 + a.pct / 100
            elif a.type == "one_time" and a.date is not None:
                if 0 <= start_i < self.H:
                    extra[start_i] += a.signed
            elif a.type == "recurring":
                d = max(a.date or self.dates[0], self.dates[0])
                while d <= self.dates[-1] and (a.end_date is None or d <= a.end_date):
                    i = self._index(d)
                    if 0 <= i < self.H:
                        extra[i] += a.signed
                    d = (
                        add_months(d, 1)
                        if a.cadence == "monthly"
                        else d
                        + dt.timedelta(
                            days={"weekly": 7, "biweekly": 14, "quarterly": 91, "yearly": 365}.get(
                                a.cadence, 30
                            )
                        )
                    )

        rec_flow, events = self._schedule(cancelled, income_mult)
        recurring_noise = np.zeros((self.n_paths, self.H))
        expected_noise = np.zeros(self.H)
        for event in events:
            amounts = self._recurring_amounts.get(event["merchant"])
            if not amounts:
                continue
            i = self._index(dt.date.fromisoformat(event["date"]))
            # Stable per-event draws keep scenario deltas independent of unrelated cancellations.
            event_seed = zlib.crc32(f"{self.seed}:{event['merchant']}:{event['date']}".encode())
            rng = np.random.default_rng(event_seed)
            scale = income_mult[i] if event["amount"] > 0 else 1.0
            recurring_noise[:, i] = (
                recurring_noise[:, i] + rng.choice(amounts, self.n_paths) * scale - event["amount"]
            )
            expected_noise[i] += float(np.mean(amounts)) * scale - event["amount"]
        variable = self.T[self._idx].astype(float)  # (P, H)
        expected_var = self._weekday_mean(self.T)
        # income changes also scale non-recurring income (freelance payouts, bonuses)
        if "income" in self.categories and not np.allclose(income_mult, 1):
            inc = self.V[self.categories.index("income")]
            variable += inc[self._idx] * (income_mult - 1)
            expected_var += self._weekday_mean(inc) * (income_mult - 1)
        for cat, mult in cat_mult.items():
            if cat not in self.categories:
                continue
            row = self.V[self.categories.index(cat)]
            variable += row[self._idx] * (mult - 1)
            expected_var += self._weekday_mean(row) * (mult - 1)
        daily = variable + rec_flow + extra + self._shocks + recurring_noise
        paths = self.balance + np.cumsum(daily, axis=1)
        shock_mean = self.shock_rate * (float(np.mean(self.shock_pool)) if self.shock_pool else 0.0)
        expected = self.balance + np.cumsum(expected_var + rec_flow + extra + shock_mean + expected_noise)
        return _RunOutput(paths, expected, rec_flow), events

    def _index(self, d: dt.date) -> int:
        return (d - self.as_of).days - 1

    # ------------------------------------------------------------------ summaries
    def summarize(
        self, out: _RunOutput, events: list[dict[str, Any]], *, series: bool = True
    ) -> dict[str, Any]:
        goals, outflow, goal_events = self._goals(out.paths, np.percentile(out.paths, 50, axis=0))
        P, expected = out.paths, out.expected
        if outflow is not None:  # spend goals: the money they were funded with leaves on the deadline
            P = P - outflow
            expected = expected - outflow.mean(axis=0)
            events = sorted(events + goal_events, key=lambda e: e["date"])
        out = _RunOutput(P, expected, out.recurring_flow)
        p10, p50, p90 = np.percentile(P, [10, 50, 90], axis=0)
        low_i = int(np.argmin(p50))
        path_min = P.min(axis=1)
        below_buffer = (self.safety_buffer > P).any(axis=1)
        risk_days = np.where(np.percentile(P, 10, axis=0) < self.safety_buffer)[0]
        res: dict[str, Any] = {
            "as_of": self.as_of.isoformat(),
            "start_balance": round(self.balance, 2),
            "horizon_days": self.H,
            "data_quality": {
                "lookback_days": self.L,
                "limited_history": self.L < LOOKBACK_DAYS,
                "balance_known": self.balance_known,
            },
            "projection_type": "balance" if self.balance_known else "cashflow_change",
            "variable_obligations": sorted(self._recurring_amounts),
            "safety_buffer": round(self.safety_buffer, 2),
            "low_point": {
                "date": self.dates[low_i].isoformat(),
                "p50": round(float(p50[low_i]), 2),
                "p10": round(float(np.percentile(path_min, 10)), 2),
            },
            "end_balance": {
                "date": self.dates[-1].isoformat(),
                "p10": round(float(p10[-1]), 2),
                "p50": round(float(p50[-1]), 2),
                "p90": round(float(p90[-1]), 2),
            },
            "prob_overdraft": round(float((path_min < 0).mean()), 3),
            "prob_below_buffer": round(float(below_buffer.mean()), 3),
            "first_risk_date": self.dates[int(risk_days[0])].isoformat() if risk_days.size else None,
            "checkpoints": self._checkpoints(p10, p50, p90),
            "goals": goals,
            "monthly": self._monthly(out, events),
            "upcoming": [e for e in events if e["date"] <= (self.as_of + dt.timedelta(days=30)).isoformat()],
        }
        days90 = min(90, self.H) - 1
        res["summary"] = {
            "net_change_90d_p50": round(float(p50[days90] - self.balance), 2),
            "avg_monthly_net": round(float((out.expected[-1] - self.balance) / self.H * 30.4), 2),
            "daily_variable_net": round(float(self._weekday_mean(self.T).mean()), 2),
        }
        if series:
            step = 1 if self.H <= 200 else 2
            res["series"] = {
                "dates": [d.isoformat() for d in self.dates[::step]],
                "p10": [round(float(x), 0) for x in p10[::step]],
                "p50": [round(float(x), 0) for x in p50[::step]],
                "p90": [round(float(x), 0) for x in p90[::step]],
                "expected": [round(float(x), 0) for x in out.expected[::step]],
            }
        if not self.balance_known:
            res.update(start_balance=None, prob_overdraft=None, prob_below_buffer=None, first_risk_date=None)
            res["required_starting_balance_p90"] = round(
                max(0.0, self.safety_buffer - float(np.percentile(path_min, 10))), 2
            )
            res["assumptions"] = [
                "Bands show net cash-flow change, not account balances.",
                "Goal odds and affordability require a verified starting balance.",
            ]
        return res

    def _checkpoints(self, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> list[dict[str, Any]]:
        out = []
        for days in (30, 60, 90, 180):
            if days <= self.H:
                i = days - 1
                out.append(
                    {
                        "days": days,
                        "date": self.dates[i].isoformat(),
                        "p10": round(float(p10[i]), 2),
                        "p50": round(float(p50[i]), 2),
                        "p90": round(float(p90[i]), 2),
                    }
                )
        return out

    def _goals(
        self, P: np.ndarray, p50: np.ndarray
    ) -> tuple[list[dict[str, Any]], np.ndarray | None, list[dict[str, Any]]]:
        """Goal odds on the pre-goal paths, plus the outflow matrix of spend goals.

        Probabilities are computed with per-path reservations (money promised to higher-priority
        goals is unavailable), which treats spend and reserve goals identically — the difference is
        only whether the money is still in the account afterwards, i.e. the returned outflow.
        """
        results = []
        outflow: np.ndarray | None = None
        goal_events: list[dict[str, Any]] = []
        reserved = np.zeros(P.shape[0])  # per-path money already promised to higher-priority goals
        reserved_med = 0.0
        avail_med = p50 - self.safety_buffer
        for g in self.goals:
            remaining = max(0.0, g.target - g.saved)
            months_left = max(0.5, (g.deadline - self.as_of).days / 30.4)
            if remaining <= 0:
                results.append(
                    {
                        "id": g.id,
                        "name": g.name,
                        "target": g.target,
                        "saved": g.saved,
                        "remaining": 0.0,
                        "deadline": g.deadline.isoformat(),
                        "probability": 1.0,
                        "status": "achieved",
                        "projected_at_deadline": g.target,
                        "shortfall": 0.0,
                        "eta": None,
                        "required_monthly": 0.0,
                        "months_left": round(months_left, 1),
                        "spend_at_deadline": g.spend_at_deadline,
                    }
                )
                continue
            di = min(self.H - 1, max(0, (g.deadline - self.as_of).days - 1))
            avail = P[:, di] - self.safety_buffer - reserved
            funded = np.clip(avail, 0, remaining)
            prob = float((funded >= remaining - 1e-6).mean())
            med_funded = float(np.clip(avail_med[di] - reserved_med, 0, remaining))
            shortfall = remaining - med_funded
            eta = self._eta(avail_med - reserved_med, remaining)
            status = "on_track" if prob >= 0.7 else ("at_risk" if prob >= 0.3 else "off_track")
            results.append(
                {
                    "id": g.id,
                    "name": g.name,
                    "target": round(g.target, 2),
                    "saved": round(g.saved, 2),
                    "remaining": round(remaining, 2),
                    "deadline": g.deadline.isoformat(),
                    "probability": round(prob, 3),
                    "status": status,
                    "projected_at_deadline": round(g.saved + med_funded, 2),
                    "shortfall": round(shortfall, 2),
                    "eta": eta.isoformat() if eta else None,
                    "required_monthly": round(shortfall / months_left, 2) if shortfall > 0 else 0.0,
                    "months_left": round(months_left, 1),
                    "spend_at_deadline": g.spend_at_deadline,
                }
            )
            reserved = reserved + funded
            reserved_med += med_funded
            if g.spend_at_deadline and (g.deadline - self.as_of).days >= 1:
                # spend what was saved for it (never more): the account keeps its buffer
                if outflow is None:
                    outflow = np.zeros_like(P)
                outflow[:, di:] += funded[:, None]
                goal_events.append(
                    {
                        "date": self.dates[di].isoformat(),
                        "merchant": f"Goal: {g.name}",
                        "amount": -round(float(np.median(funded)), 2),
                        "kind": "goal_spend",
                        "category": "goal",
                    }
                )
        return results, outflow, goal_events

    def _eta(self, avail: np.ndarray, remaining: float) -> dt.date | None:
        # first day from which the goal stays funded (salary-day spikes don't count)
        sustained = np.minimum.accumulate(avail[::-1])[::-1] >= remaining
        hit = np.where(sustained)[0]
        if hit.size:
            return self.dates[int(hit[0])]
        tail = min(60, self.H - 1)
        slope = (avail[-1] - avail[-1 - tail]) / tail if tail > 0 else 0.0
        if slope <= 0:
            return None
        days = (remaining - avail[-1]) / slope
        if days > 3650:
            return None
        return self.dates[-1] + dt.timedelta(days=int(math.ceil(days)))

    def _monthly(self, out: _RunOutput, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expected next-3-calendar-month cash-flow breakdown."""
        months: dict[str, dict[str, Any]] = {}
        for e in events:
            k = e["date"][:7]
            m = months.setdefault(
                k, {"month": k, "income": 0.0, "commitments": 0.0, "variable_by_category": {}}
            )
            if e["amount"] > 0:
                m["income"] += e["amount"]
            else:
                m["commitments"] += -e["amount"]
        for ci, cat in enumerate(self.categories):
            daily = self._weekday_mean(self.V[ci])
            per_month: dict[str, float] = defaultdict(float)
            for d, v in zip(self.dates, daily, strict=False):
                per_month[month_key(d)] += v
            for k, v in per_month.items():
                m = months.setdefault(
                    k, {"month": k, "income": 0.0, "commitments": 0.0, "variable_by_category": {}}
                )
                if abs(v) >= 1:
                    m["variable_by_category"][cat] = round(-v, 2)
        out_list = []
        projected_net: dict[str, float] = defaultdict(float)
        previous = self.balance
        for date, value in zip(self.dates, out.expected, strict=True):
            key = month_key(date)
            months.setdefault(
                key, {"month": key, "income": 0.0, "commitments": 0.0, "variable_by_category": {}}
            )
            projected_net[key] += float(value) - previous
            previous = float(value)
        for k in sorted(months)[:4]:
            m = months[k]
            var_in = sum(-v for v in m["variable_by_category"].values() if v < 0)
            var_out = sum(v for v in m["variable_by_category"].values() if v > 0)
            m["income"] = round(m["income"] + var_in, 2)
            m["commitments"] = round(m["commitments"], 2)
            m["variable_spend"] = round(var_out, 2)
            component_net = m["income"] - m["commitments"] - var_out
            m["other_projected_cashflow"] = round(projected_net[k] - component_net, 2)
            m["net"] = round(projected_net[k], 2)
            m["partial"] = k == month_key(self.dates[0]) or k == month_key(self.dates[-1])
            out_list.append(m)
        return out_list

    def category_monthly_rates(self) -> dict[str, float]:
        """Expected monthly spend per variable category under current behaviour."""
        out = {}
        for ci, cat in enumerate(self.categories):
            v = -float(self._weekday_mean(self.V[ci]).mean()) * 30.4
            if v > 0:
                out[cat] = round(v, 2)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _per_month(cadence: str) -> float:
    return {"weekly": 52 / 12, "biweekly": 26 / 12, "monthly": 1.0, "quarterly": 1 / 3, "yearly": 1 / 12}.get(
        cadence, 1.0
    )


def _advance(d: dt.date, r: RecurringCandidate) -> dt.date:
    if r.cadence == "monthly":
        return add_months(d, 1)
    if r.cadence == "quarterly":
        return add_months(d, 3)
    if r.cadence == "yearly":
        return add_months(d, 12)
    return d + dt.timedelta(days=round(r.interval_days) or 7)


# ====================================================================== agent

SYSTEM_PROMPT = """You are the Forecaster agent of FinPilot. You receive a Monte-Carlo cash-flow projection for ONE user
(every number is already computed and correct). Explain it in 2-3 plain sentences, second person:
1. where the balance is heading if current behaviour continues (a checkpoint's P50 with its P10–P90 range);
2. the riskiest moment (the low point date and value, and overdraft probability when known);
3. what it means for the most at-risk goal (probability, shortfall, ETA), or that the goals are on track.
Check projection_type: "cashflow_change" means the starting balance is unknown — then describe changes and the
required starting cash, never an absolute balance, overdraft risk or goal odds. Forecasts are conditional estimates
from the user's own history, not guarantees; mention short history or variable obligations when data_quality says so.
Copy figures exactly as they appear in the input (keep decimals) — never round or compute new numbers.
Currency: {currency}.
Respond with JSON only: {{"narrative": "..."}}"""


class ForecasterAgent(BaseAgent):
    name = "forecaster"
    role = "Projects the cash trajectory (P10/P50/P90), goal probabilities, and runs what-if simulations"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, *args: Any, n_paths: int = 400, horizon_days: int = 180, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.n_paths = n_paths
        self.horizon_days = horizon_days

    def engine(
        self,
        txns: list[TxnLite],
        recurring: list[RecurringCandidate],
        *,
        balance: float,
        as_of: dt.date,
        goals: list[GoalInput],
        exclude_ids: set[int],
        safety_buffer: float | None = None,
        horizon_days: int | None = None,
        seed: int = 42,
        balance_known: bool = True,
    ) -> ForecastEngine:
        return ForecastEngine(
            txns,
            recurring,
            balance=balance,
            as_of=as_of,
            goals=goals,
            exclude_ids=exclude_ids,
            horizon_days=horizon_days or self.horizon_days,
            n_paths=self.n_paths,
            seed=seed,
            safety_buffer=safety_buffer,
            balance_known=balance_known,
        )

    def forecast(self, eng: ForecastEngine, *, currency: str = "INR", narrate: bool = True) -> dict[str, Any]:
        out, events = eng.run()
        res = eng.summarize(out, events)
        res["category_monthly"] = eng.category_monthly_rates()
        res["narrative"], res["narrative_model"] = (
            self._narrate(res, currency) if narrate else (self._template(res, currency), "rules")
        )
        return res

    # ------------------------------------------------------------------ what-if
    def simulate(
        self,
        eng: ForecastEngine,
        adjustments: list[Adjustment],
        *,
        currency: str = "INR",
        search_wait: bool = True,
    ) -> dict[str, Any]:
        base_out, base_events = eng.run()
        base = eng.summarize(base_out, base_events)
        scen_out, scen_events = eng.run(adjustments)
        scen = eng.summarize(scen_out, scen_events)
        verdict, reasons = self._verdict(base, scen, eng.safety_buffer, currency)
        goals_delta = []
        for gb, gs in zip(base["goals"], scen["goals"], strict=False):
            delay = None
            if gb.get("eta") and gs.get("eta"):
                delay = (dt.date.fromisoformat(gs["eta"]) - dt.date.fromisoformat(gb["eta"])).days
            goals_delta.append(
                {
                    "name": gb["name"],
                    "prob_before": gb["probability"],
                    "prob_after": gs["probability"],
                    "eta_before": gb["eta"],
                    "eta_after": gs["eta"],
                    "delay_days": delay,
                    "shortfall_before": gb["shortfall"],
                    "shortfall_after": gs["shortfall"],
                }
            )
        result: dict[str, Any] = {
            "adjustments": [
                dict(
                    asdict(a),
                    date=a.date.isoformat() if a.date else None,
                    end_date=a.end_date.isoformat() if a.end_date else None,
                    description=a.describe(currency),
                )
                for a in adjustments
            ],
            "verdict": verdict,
            "reasons": reasons,
            "baseline": base,
            "scenario": scen,
            "delta": {
                "end_balance_p50": round(scen["end_balance"]["p50"] - base["end_balance"]["p50"], 2),
                "low_point_p50": round(scen["low_point"]["p50"] - base["low_point"]["p50"], 2),
                "prob_overdraft": round(scen["prob_overdraft"] - base["prob_overdraft"], 3)
                if eng.balance_known
                else None,
                "prob_below_buffer": round(scen["prob_below_buffer"] - base["prob_below_buffer"], 3)
                if eng.balance_known
                else None,
                "goals": goals_delta,
            },
            "suggestions": [],
        }
        if eng.balance_known and verdict != "affordable":
            result["suggestions"] = self._suggestions(eng, adjustments, base, currency, search_wait)
        return result

    @staticmethod
    def _verdict(
        base: dict[str, Any], scen: dict[str, Any], buffer: float, currency: str
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if scen.get("projection_type") == "cashflow_change":
            return "needs_balance", [
                "The simulation measures cash-flow changes. A verified starting balance is needed to decide affordability.",
                "Goal funding is excluded until a verified balance is available.",
                f"Under the modeled cash flows, starting cash of {format_money(scen['required_starting_balance_p90'], currency, 2)} would preserve the buffer in 90% of paths; this is a model estimate.",
            ]
        low10, low50 = scen["low_point"]["p10"], scen["low_point"]["p50"]
        verdict = "affordable"
        if scen["prob_overdraft"] >= 0.10:
            verdict = "not_affordable"
            reasons.append(
                f"{scen['prob_overdraft'] * 100:.0f}% chance your balance goes negative "
                f"(worst-case low {format_money(low10, currency, 0)})."
            )
        elif low50 < buffer:
            verdict = "stretch"
            reasons.append(
                f"Your balance would dip to about {format_money(low50, currency, 0)} around {scen['low_point']['date']}, "
                f"below your {format_money(buffer, currency, 0)} safety buffer."
            )
        for gb, gs in zip(base["goals"], scen["goals"], strict=False):
            drop = gb["probability"] - gs["probability"]
            delay = None
            if gb.get("eta") and gs.get("eta"):
                delay = (dt.date.fromisoformat(gs["eta"]) - dt.date.fromisoformat(gb["eta"])).days
            elif gb.get("eta") and not gs.get("eta"):
                delay = 9999
            if drop >= 0.15 or (gb["probability"] >= 0.7 > gs["probability"]):
                if verdict == "affordable":
                    verdict = "stretch"
                reasons.append(
                    f"'{gb['name']}' odds fall from {gb['probability'] * 100:.0f}% to "
                    f"{gs['probability'] * 100:.0f}%."
                )
            elif delay is not None and delay >= 30 and gb["status"] != "achieved":
                if verdict == "affordable":
                    verdict = "stretch"
                if delay >= 9999:
                    reasons.append(f"'{gb['name']}' would no longer be reached within the forecast horizon.")
                else:
                    reasons.append(
                        f"Pushes '{gb['name']}' back by about {delay // 7} weeks "
                        f"(ETA {gb['eta']} → {gs['eta']})."
                    )
        if verdict == "affordable":
            reasons.append(
                f"Lowest projected balance stays around {format_money(low50, currency, 0)} (above your "
                f"{format_money(buffer, currency, 0)} buffer) and goal odds hold."
            )
        return verdict, reasons

    def _suggestions(
        self,
        eng: ForecastEngine,
        adjustments: list[Adjustment],
        base: dict[str, Any],
        currency: str,
        search_wait: bool,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        one_offs = [a for a in adjustments if a.type == "one_time" and a.kind == "expense" and a.date]
        if search_wait and one_offs and len(adjustments) == len(one_offs):
            for weeks in range(1, 27):
                shifted = [
                    Adjustment(**{**asdict(a), "date": a.date + dt.timedelta(weeks=weeks)}) for a in one_offs
                ]
                if shifted[0].date > eng.dates[-1]:
                    break
                o, ev = eng.run(shifted)
                s = eng.summarize(o, ev, series=False)
                v, _ = self._verdict(base, s, eng.safety_buffer, currency)
                if v == "affordable":
                    out.append(
                        {
                            "type": "wait",
                            "date": shifted[0].date.isoformat(),
                            "weeks": weeks,
                            "text": f"Wait until {shifted[0].date:%d %b %Y} ({weeks} weeks later) and it fits "
                            f"without touching your buffer or goals.",
                        }
                    )
                    break
        cost = sum(abs(a.amount) for a in adjustments if a.kind == "expense" and a.type == "one_time")
        if cost > 0:
            rates = eng.category_monthly_rates()
            disc = [(c, v) for c, v in rates.items() if c in DISCRETIONARY and v > 500][:3]
            pool = sum(v for _, v in disc)
            if pool > 0:
                # a sustainable cut is ~35% of discretionary spend; stretch to 60% only if 6 months isn't enough
                pct = 0.35
                months = math.ceil(cost / (pool * pct))
                if months > 6:
                    months, pct = 6, min(0.6, cost / (6 * pool))
                plan = [
                    {"category": c, "label": label(c), "current_monthly": v, "cut": round(v * pct, 0)}
                    for c, v in disc
                ]
                monthly = sum(p["cut"] for p in plan)
                funded = min(cost, monthly * months)
                parts = ", ".join(
                    f"{p['label']} ({format_money(p['current_monthly'], currency, 0)} → "
                    f"{format_money(p['current_monthly'] - p['cut'], currency, 0)})"
                    for p in plan
                )
                text = (
                    f"Cut {parts} — {pct * 100:.0f}% less, saving {format_money(monthly, currency, 0)}/month — for {months} "
                    f"month{'s' if months > 1 else ''} to fund it"
                )
                text += (
                    "."
                    if funded >= cost - 1
                    else f"; that covers {format_money(funded, currency, 0)} of the {format_money(cost, currency, 0)}."
                )
                out.append(
                    {
                        "type": "cut",
                        "months": months,
                        "pct": round(pct * 100),
                        "monthly_saving": monthly,
                        "covered": round(funded, 0),
                        "plan": plan,
                        "text": text,
                    }
                )
        return out

    # ------------------------------------------------------------------ narrative
    @staticmethod
    def _template(res: dict[str, Any], currency: str) -> str:
        cps = {c["days"]: c for c in res["checkpoints"]}
        cp = cps.get(90) or (res["checkpoints"][-1] if res["checkpoints"] else None)
        parts = []
        if res.get("projection_type") == "cashflow_change":
            return (
                (
                    f"Your starting balance is unverified. Estimated net cash-flow change over {cp['days']} days is "
                    f"{format_money(cp['p50'], currency, 2)} (P10 {format_money(cp['p10'], currency, 2)}; P90 {format_money(cp['p90'], currency, 2)}). "
                    "These are changes, not account balances; goal odds and affordability need a verified balance."
                )
                if cp
                else "A verified starting balance is needed for balance and goal projections."
            )
        if cp:
            parts.append(
                f"If you keep spending like this, your balance should be around {format_money(cp['p50'], currency, 0)} in "
                f"{cp['days']} days (likely range {format_money(cp['p10'], currency, 0)}–{format_money(cp['p90'], currency, 0)})."
            )
        lp = res["low_point"]
        parts.append(
            f"The tightest point is around {lp['date']} at about {format_money(lp['p50'], currency, 0)}."
        )
        risky = [g for g in res["goals"] if g["status"] in ("at_risk", "off_track")]
        if risky:
            g = risky[0]
            parts.append(
                f"'{g['name']}' is {g['probability'] * 100:.0f}% likely by {g['deadline']} — about "
                f"{format_money(g['shortfall'], currency, 0)} short at this pace."
            )
        elif res["goals"]:
            parts.append("Your goals are on track.")
        return " ".join(parts)

    def _narrate(self, res: dict[str, Any], currency: str) -> tuple[str, str]:
        fallback = self._template(res, currency)
        if not self.llm_available:
            return fallback, "rules"
        facts = {
            k: res[k]
            for k in (
                "start_balance",
                "safety_buffer",
                "low_point",
                "end_balance",
                "prob_overdraft",
                "checkpoints",
                "summary",
                "projection_type",
                "data_quality",
                "variable_obligations",
            )
        }
        facts["goals"] = [
            {
                k: g[k]
                for k in ("name", "target", "deadline", "probability", "shortfall", "eta", "required_monthly")
            }
            for g in res["goals"]
        ]
        self.system_prompt = SYSTEM_PROMPT.format(currency=currency)
        facts["required_starting_balance_p90"] = res.get("required_starting_balance_p90")
        data, label_, _ = self.safe_ask_json(facts, max_tokens=1000, temperature=0.1)
        text = data.get("narrative") if isinstance(data, dict) else None
        if not text or not check_grounding(str(text), facts, strict=True)[0]:
            return fallback, "rules"
        (verdict,) = self.validate([("forecast narrative", facts, str(text))])
        if not verdict.ok:
            return fallback, "rules"
        return str(text), label_ or "llm"
