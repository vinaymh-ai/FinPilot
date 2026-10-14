"""Read-only finance tools over one user's real data.

Shared by the conversational agent (as LLM function-calling tools *and* by its deterministic
fallback) and by the REST API, so every number a user sees comes from the same code path.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal
from functools import cached_property
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from finpilot.agents.forecaster import Adjustment, ForecastEngine, ForecasterAgent
from finpilot.config import get_settings
from finpilot.domain.normalize import merchant_matches, normalize_text
from finpilot.domain.recurring import monthly_equivalent
from finpilot.domain.stats import add_months, month_key
from finpilot.domain.taxonomy import BY_ID, CATEGORIES, NON_SPEND, is_spending_flow, label
from finpilot.services import repository as repo

SYNONYMS = {
    "food": ["food_delivery", "dining"],
    "eating out": ["dining"],
    "restaurants": ["dining"],
    "restaurant": ["dining"],
    "delivery": ["food_delivery"],
    "takeout": ["food_delivery"],
    "groceries": ["groceries"],
    "grocery": ["groceries"],
    "rent": ["rent"],
    "bills": ["utilities"],
    "utilities": ["utilities"],
    "electricity": ["utilities"],
    "transport": ["transport"],
    "travel": ["travel"],
    "cab": ["transport"],
    "cabs": ["transport"],
    "fuel": ["transport"],
    "petrol": ["transport"],
    "shopping": ["shopping"],
    "clothes": ["shopping"],
    "subscriptions": ["subscriptions"],
    "subscription": ["subscriptions"],
    "entertainment": ["entertainment"],
    "movies": ["entertainment"],
    "health": ["health"],
    "medical": ["health"],
    "gym": ["health"],
    "education": ["education"],
    "emi": ["loans"],
    "loans": ["loans"],
    "credit card": ["credit_card_payment"],
    "card repayment": ["credit_card_payment"],
    "insurance": ["insurance"],
    "investments": ["investments"],
    "sip": ["investments"],
    "salon": ["personal_care"],
    "personal care": ["personal_care"],
    "cash": ["cash"],
    "fees": ["fees"],
    "gifts": ["gifts_donations"],
    "donations": ["gifts_donations"],
    "income": ["income"],
    "salary": ["income"],
    "transfers": ["transfers"],
}


def resolve_categories(text: str | None) -> list[str]:
    if not text:
        return []
    t = text.strip().lower()
    if t in BY_ID:
        return [t]
    if t in SYNONYMS:
        return SYNONYMS[t]
    for c in CATEGORIES:
        if t == c.label.lower() or t.replace(" ", "_") == c.id:
            return [c.id]
    for key, cats in SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}\b", t):
            return cats
    return []


class FinanceToolkit:
    def __init__(self, session: Session, user_id: int = repo.DEFAULT_USER_ID):
        self.session = session
        self.user_id = user_id
        self.user = repo.get_or_create_user(session, user_id)
        self.currency = self.user.currency or get_settings().currency

    # ------------------------------------------------------------------ cached data
    @cached_property
    def rows(self):
        return repo.load_transactions(self.session, self.user_id)

    @cached_property
    def txns(self):
        return [repo.to_lite(t) for t in self.rows]

    @cached_property
    def recurring(self):
        return repo.recurring_candidates(self.session, self.user_id)

    @cached_property
    def goals(self):
        return repo.goal_inputs(self.session, self.user_id)

    @cached_property
    def balance_info(self) -> tuple[float, dt.date]:
        return repo.current_balance(self.session, self.user)

    @property
    def as_of(self) -> dt.date:
        return self.balance_info[1]

    @property
    def today(self) -> dt.date:
        return dt.datetime.now(ZoneInfo(get_settings().timezone)).date()

    def engine(self, seed: int = 42) -> ForecastEngine:
        s = get_settings()
        return ForecasterAgent(use_llm=False, n_paths=s.monte_carlo_paths).engine(
            self.txns,
            self.recurring,
            balance=self.balance_info[0],
            as_of=self.as_of,
            goals=self.goals,
            exclude_ids={t.id for t in self.rows if t.is_anomaly},
            safety_buffer=self.user.safety_buffer if self.user.safety_buffer is not None else s.safety_buffer,
            horizon_days=s.forecast_horizon_days,
            seed=seed,
            balance_known=self.data_quality()["balance_known"],
        )

    def money(self, x: float) -> str:
        sym = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(self.currency, "")
        digits = 2 if round(abs(x), 2) != round(abs(x)) else 0
        return f"{'-' if x < 0 else ''}{sym}{abs(x):,.{digits}f}"

    def data_quality(self) -> dict[str, Any]:
        dates = [t.date for t in self.rows]
        review = [
            t.id
            for t in self.rows
            if t.category in (None, "other")
            or (t.category_confidence is not None and t.category_confidence < 0.7)
        ]
        days = (max(dates) - min(dates)).days + 1 if dates else 0
        return {
            "transactions": len(dates),
            "balance_known": self.user.balance_source in {"statement", "user", "demo"},
            "start": min(dates).isoformat() if dates else None,
            "end": max(dates).isoformat() if dates else None,
            "history_days": days,
            "needs_review": len(review),
            "review_transaction_ids": review[:100],
            "forecast_reliability": "limited" if days < 90 or review else "historical_estimate",
        }

    # ------------------------------------------------------------------ period helpers
    def period_range(
        self, period: str | None = None, start_date: str | None = None, end_date: str | None = None
    ) -> tuple[dt.date, dt.date, str]:
        a = self.today
        if start_date or end_date:
            s = dt.date.fromisoformat(start_date) if start_date else a - dt.timedelta(days=30)
            e = dt.date.fromisoformat(end_date) if end_date else a
            if s > e:
                raise ValueError("start_date must be on or before end_date")
            return s, e, f"{s.isoformat()} to {e.isoformat()}"
        p = (period or "last_30_days").lower().replace(" ", "_")
        if p in {"all_time", "all", "entire_history"}:
            return min((t.date for t in self.txns), default=a), self.as_of, "all available history"
        if p == "this_month":
            return a.replace(day=1), a, f"{a:%B %Y} (to date)"
        if p == "last_month":
            s = add_months(a.replace(day=1), -1)
            return s, a.replace(day=1) - dt.timedelta(days=1), f"{s:%B %Y}"
        if p in {"today", "yesterday"}:
            day = a - dt.timedelta(days=p == "yesterday")
            return day, day, p
        if p in {"this_week", "last_week"}:
            monday = a - dt.timedelta(days=a.weekday())
            if p == "last_week":
                return monday - dt.timedelta(days=7), monday - dt.timedelta(days=1), "last calendar week"
            return monday, a, "this calendar week"
        if p == "last_year":
            return dt.date(a.year - 1, 1, 1), dt.date(a.year - 1, 12, 31), str(a.year - 1)
        if re.fullmatch(r"\d{4}-\d{2}", p):
            s = dt.date.fromisoformat(p + "-01")
            return s, add_months(s, 1) - dt.timedelta(days=1), f"{s:%B %Y}"
        m = re.fullmatch(r"last_(\d+)_days?", p)
        days = int(m.group(1)) if m else 30
        if p == "this_year":
            return a.replace(month=1, day=1), a, f"{a.year} (to date)"
        if not m:
            raise ValueError(f"Unsupported period: {period}")
        if not 1 <= days <= 36500:
            raise ValueError("Period must be between 1 and 36500 days")
        return a - dt.timedelta(days=days - 1), a, f"last {days} days"

    # ------------------------------------------------------------------ tools
    def get_overview(self) -> dict[str, Any]:
        bal, as_of = self.balance_info
        start, end, _ = self.period_range("last_30_days")
        recent = [t for t in self.txns if start <= t.date <= end]
        quality = self.data_quality()
        outflows = [t for t in recent if t.amount < 0]
        spending_outflows = [t for t in outflows if is_spending_flow(t.category, t.amount)]
        income = sum(t.amount for t in recent if t.category == "income")
        spend = sum(-t.amount for t in recent if is_spending_flow(t.category, t.amount))
        by_cat: dict[str, float] = defaultdict(float)
        for t in recent:
            if is_spending_flow(t.category, t.amount):
                by_cat[t.category] += -t.amount
        subs = [r for r in self.recurring if r.kind == "subscription" and r.status == "active"]
        return {
            "as_of": as_of.isoformat(),
            "currency": self.currency,
            "balance": bal if quality["balance_known"] else None,
            "last_30_days": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "income": round(income, 2),
                "spend": round(spend, 2),
                "cash_outflows": round(sum(-t.amount for t in outflows), 2),
                "cash_outflow_count": len(outflows),
                "spending_outflow_count": len(spending_outflows),
                "excluded_outflows": round(sum(-t.amount for t in outflows if t.category in NON_SPEND), 2),
                "spending_definition": "Spending excludes transfers and investments. Cash outflows include every recorded debit.",
                "savings_rate_pct": round((income - spend) / income * 100, 1) if income else None,
            },
            "top_categories_30d": [
                {"category": label(c), "amount": round(v, 2)}
                for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])[:6]
            ],
            "subscriptions": {
                "count": len(subs),
                "monthly_cost": round(
                    sum(monthly_equivalent(abs(r.typical_amount), r.cadence) for r in subs), 2
                ),
            },
            "transactions": len(self.txns),
            "data_quality": quality,
            "goals": [
                {"name": g.name, "target": g.target, "saved": g.saved, "deadline": g.deadline.isoformat()}
                for g in self.goals
            ],
        }

    def get_spending(
        self,
        category: str | None = None,
        merchant: str | None = None,
        period: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        s, e, desc = self.period_range(period, start_date, end_date)
        cats = resolve_categories(category)
        if category and not cats:
            raise ValueError(f"Unknown category: {category}")
        m_q = (merchant or "").strip()
        rows = [t for t in self.txns if s <= t.date <= e and t.amount < 0]
        if cats:
            rows = [t for t in rows if t.category in cats]
        elif not m_q:
            rows = [t for t in rows if t.category not in NON_SPEND]
        if m_q:
            rows = [t for t in rows if merchant_matches(m_q, t.merchant, t.description)]
        total = sum((-Decimal(str(t.amount)) for t in rows), Decimal("0"))
        refunds = [
            t
            for t in self.txns
            if s <= t.date <= e
            and t.amount > 0
            and is_spending_flow(t.category, t.amount)
            and (not cats or t.category in cats)
            and (not m_q or merchant_matches(m_q, t.merchant, t.description))
        ]
        refund_total = sum((Decimal(str(t.amount)) for t in refunds), Decimal("0"))
        merchants = Counter()
        by_cat: dict[str, float] = defaultdict(float)
        monthly: dict[str, float] = defaultdict(float)
        for t in rows:
            merchants[t.merchant] += -t.amount
            by_cat[t.category] += -t.amount
            monthly[month_key(t.date)] += -t.amount
        days = (e - s).days + 1
        return {
            "period": desc,
            "start": s.isoformat(),
            "end": e.isoformat(),
            "filter": {"categories": [label(c) for c in cats], "merchant": m_q or None},
            "total": float(round(total, 2)),
            "refunds": float(round(refund_total, 2)),
            "net_total": float(round(total - refund_total, 2)),
            "count": len(rows),
            "avg_per_txn": float(round(total / len(rows), 2)) if rows else 0.0,
            "per_month_rate": round(float(total) / days * 30.4, 2) if days else 0.0,
            "by_category": {label(c): round(v, 2) for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
            "top_merchants": [{"merchant": m, "amount": round(a, 2)} for m, a in merchants.most_common(5)],
            "monthly": {k: round(v, 2) for k, v in sorted(monthly.items())},
            "transaction_ids": [t.id for t in rows],
            "matched_merchants": sorted({t.merchant for t in rows}),
            "data_as_of": self.as_of.isoformat(),
        }

    def rank_spending(
        self,
        group_by: str = "transaction",
        period: str = "all_time",
        limit: int = 5,
        category: str | None = None,
        merchant: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Rank real expenses; transfers and investments are not consumption expenses."""
        if group_by not in {"transaction", "category", "merchant"}:
            raise ValueError("group_by must be transaction, category or merchant")
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("limit must be an integer between 1 and 50")
        totals = self.get_spending(category, merchant, period, start_date, end_date)
        ids = set(totals["transaction_ids"])
        rows = [t for t in self.txns if t.id in ids and t.category not in NON_SPEND]
        ranked = []
        if group_by == "transaction":
            ranked = [
                {
                    "id": t.id,
                    "date": t.date.isoformat(),
                    "merchant": t.merchant,
                    "category": label(t.category),
                    "amount": round(-t.amount, 2),
                }
                for t in sorted(rows, key=lambda t: (-abs(t.amount), t.date, t.id))
            ]
        else:
            groups = defaultdict(list)
            for t in rows:
                groups[label(t.category) if group_by == "category" else t.merchant].append(t)
            ranked = [
                {
                    "name": name,
                    "amount": float(sum((-Decimal(str(t.amount)) for t in items), Decimal(0))),
                    "count": len(items),
                    "transaction_ids": [t.id for t in items],
                }
                for name, items in groups.items()
            ]
            ranked.sort(key=lambda r: (-r["amount"], r["name"]))
        return {
            "group_by": group_by,
            "start": totals["start"],
            "end": totals["end"],
            "ranked": ranked[:limit],
            "eligible_count": len(rows),
            "truncated": len(ranked) > limit,
            "definition": "Recorded expense debits, including debt/card repayments; excluding transfers and investments.",
            "data_as_of": self.as_of.isoformat(),
        }

    def search_transactions(
        self,
        query: str | None = None,
        category: str | None = None,
        min_amount: float | None = None,
        limit: int = 10,
        period: str | None = "all_time",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        cats = resolve_categories(category)
        if category and not cats:
            raise ValueError(f"Unknown category: {category}")
        if min_amount is not None and (not math.isfinite(min_amount) or min_amount < 0):
            raise ValueError("min_amount must be a finite nonnegative number")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        s, e, desc = self.period_range(period, start_date, end_date)
        q = (query or "").strip()
        rows = [
            t
            for t in self.txns
            if s <= t.date <= e
            and (not q or merchant_matches(q, t.merchant, t.description))
            and (not cats or t.category in cats)
            and (min_amount is None or abs(t.amount) >= min_amount)
        ]
        total_count = len(rows)
        possible: dict[str, list[int]] = {}
        tokens = normalize_text(q).split()
        if q and not rows and len(tokens) >= 2:
            for t in self.txns:
                if not s <= t.date <= e or (cats and t.category not in cats):
                    continue
                for fragment in [t.merchant, *t.description.split("/")]:
                    words = normalize_text(fragment).split()
                    if len(words) != len(tokens) or words[0] != tokens[0]:
                        continue
                    if all(
                        a == b or (len(b) >= 4 and a.startswith(b))
                        for a, b in zip(tokens, words, strict=True)
                    ):
                        ids = possible.setdefault(fragment.strip(), [])
                        if t.id not in ids:
                            ids.append(t.id)
        rows = sorted(rows, key=lambda t: (t.date, t.time or dt.time(), t.id), reverse=True)[: min(limit, 50)]
        return {
            "count": total_count,
            "returned_count": len(rows),
            "truncated": total_count > len(rows),
            "period": desc,
            "start": s.isoformat(),
            "end": e.isoformat(),
            "filter": {"query": q or None, "categories": cats, "min_amount": min_amount},
            "covers_all_recorded_transactions_in_period": len(rows)
            == sum(s <= t.date <= e for t in self.txns),
            "possible_matches": [
                {"statement_name": name, "transaction_ids": ids} for name, ids in list(possible.items())[:5]
            ],
            "transactions": [
                {
                    "id": t.id,
                    "date": t.date.isoformat(),
                    "merchant": t.merchant,
                    "amount": t.amount,
                    "category": label(t.category),
                    "description": t.description,
                }
                for t in rows
            ],
        }

    def get_subscriptions(self) -> dict[str, Any]:
        from finpilot.models import RecurringSeries

        rows = self.session.query(RecurringSeries).filter_by(user_id=self.user_id, kind="subscription").all()
        linked = {t.recurring_id for t in self.rows if t.recurring_id is not None}
        rows = [r for r in rows if r.id in linked]
        items = [
            {
                "merchant": r.merchant,
                "amount": abs(float(r.typical_amount)),
                "cadence": r.cadence,
                "monthly_cost": round(monthly_equivalent(abs(float(r.typical_amount)), r.cadence), 2),
                "status": r.status,
                "next_charge": r.next_expected.isoformat(),
                "usage": r.usage,
                "price_change_pct": r.price_change_pct,
                "usage_signal": r.usage_signal,
            }
            for r in rows
        ]
        active = [i for i in items if i["status"] == "active"]
        return {
            "count_active": len(active),
            "monthly_total": round(sum(i["monthly_cost"] for i in active), 2),
            "items": sorted(items, key=lambda i: -i["monthly_cost"]),
        }

    def get_forecast(self, days_ahead: int | None = None, date: str | None = None) -> dict[str, Any]:
        eng = self.engine()
        out, events = eng.run()
        res = eng.summarize(out, events, series=False)
        target = None
        if date:
            target = dt.date.fromisoformat(date)
        elif days_ahead:
            target = self.as_of + dt.timedelta(days=int(days_ahead))
        at = None
        if target:
            if not self.as_of < target <= eng.dates[-1]:
                raise ValueError(f"Forecast date must be after {self.as_of} and on or before {eng.dates[-1]}")
            i = (target - self.as_of).days - 1
            import numpy as np

            p10, p50, p90 = np.percentile(out.paths[:, i], [10, 50, 90])
            at = {
                "date": eng.dates[i].isoformat(),
                "p10": round(float(p10), 2),
                "p50": round(float(p50), 2),
                "p90": round(float(p90), 2),
            }
        return {
            "start_balance": res["start_balance"],
            "projection_type": res["projection_type"],
            "required_starting_balance_p90": res.get("required_starting_balance_p90"),
            "as_of": res["as_of"],
            "at_date": at,
            "checkpoints": res["checkpoints"],
            "low_point": res["low_point"],
            "end_balance": res["end_balance"],
            "prob_overdraft": res["prob_overdraft"],
            "safety_buffer": res["safety_buffer"],
            "goals": res["goals"],
            "avg_monthly_net": res["summary"]["avg_monthly_net"],
            "upcoming_bills": res["upcoming"][:8],
            "data_quality": self.data_quality(),
        }

    def simulate(self, adjustments: list[dict[str, Any]], full: bool = False) -> dict[str, Any]:
        adjs = [parse_adjustment(a, self.as_of) for a in adjustments or []]
        if not adjs:
            return {"error": "no valid adjustments"}
        res = ForecasterAgent(use_llm=False).simulate(self.engine(), adjs, currency=self.currency)
        if full:
            return res
        return {
            "verdict": res["verdict"],
            "projection_type": res["scenario"]["projection_type"],
            "required_starting_balance_p90": res["scenario"].get("required_starting_balance_p90"),
            "delta": res["delta"],
            "reasons": res["reasons"],
            "adjustments": [a["description"] for a in res["adjustments"]],
            "baseline": {
                "low_point": res["baseline"]["low_point"],
                "end_balance": res["baseline"]["end_balance"],
            },
            "scenario": {
                "low_point": res["scenario"]["low_point"],
                "end_balance": res["scenario"]["end_balance"],
                "prob_overdraft": res["scenario"]["prob_overdraft"],
            },
            "goals": res["delta"]["goals"],
            "suggestions": [s["text"] for s in res["suggestions"]],
            "safety_buffer": res["baseline"]["safety_buffer"],
        }

    def get_goals(self) -> dict[str, Any]:
        if not self.data_quality()["balance_known"]:
            return {
                "goals": [
                    {
                        "name": g.name,
                        "target": g.target,
                        "saved": g.saved,
                        "remaining": round(max(0, g.target - g.saved), 2),
                        "deadline": str(g.deadline),
                        "probability": None,
                        "status": "achieved" if g.saved >= g.target else "needs_balance",
                    }
                    for g in self.goals
                ],
                "needs_balance": True,
                "explanation": "Recorded goal progress is available; forecast odds need a verified balance.",
            }
        f = repo.latest_forecast(self.session, self.user_id)
        if f is not None:
            return {"goals": f.payload.get("goals", []), "as_of": f.payload.get("as_of")}
        return {"goals": self.get_forecast()["goals"]}

    def get_recommendations(self) -> dict[str, Any]:
        recs = repo.latest_recommendations(self.session, self.user_id)
        return {
            "recommendations": [
                {
                    "title": r.title,
                    "message": r.message,
                    "action": r.action,
                    "monthly_impact": r.monthly_impact,
                    "trigger": r.trigger,
                    "status": r.status,
                }
                for r in recs[:6]
            ]
        }

    def get_insights(self) -> dict[str, Any]:
        return {
            "insights": [
                {"kind": i.kind, "severity": i.severity, "title": i.title, "detail": i.detail}
                for i in repo.latest_insights(self.session, self.user_id)[:10]
            ]
        }

    def estimate_trip_cost(self) -> dict[str, Any]:
        """Typical trip cost from the user's own travel history (clusters of travel spend within 10 days)."""
        travel = sorted(
            [t for t in self.txns if t.category == "travel" and t.amount < 0], key=lambda t: t.date
        )
        trips: list[float] = []
        last: dt.date | None = None
        for t in travel:
            if trips and last is not None and (t.date - last).days <= 10:
                trips[-1] += -t.amount
            else:
                trips.append(-t.amount)
            last = t.date
        trips = [x for x in trips if x >= 2500]  # ignore bus tickets / day outings
        if trips:
            # a trip "like your last real one": the biggest past trip, plus ~30% for on-trip spending
            est = max(trips) * 1.3
            basis = f"your biggest past trip ({self.money(max(trips))}) + 30% for on-trip spending"
        else:
            return {
                "estimate": None,
                "basis": "No past trips found; supply a budget rather than guessing a cost.",
                "past_trips": [],
            }
        return {"estimate": round(est, -2), "basis": basis, "past_trips": [round(x, 2) for x in trips]}

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return {"error": f"unknown tool {name}"}
        if not isinstance(args, dict):
            return {"error": "tool arguments must be a JSON object"}
        try:
            return fn(**(args or {}))
        except TypeError as exc:
            return {"error": f"bad arguments: {exc}"}
        except ValueError as exc:
            return {"error": str(exc)}


def parse_adjustment(a: dict[str, Any], as_of: dt.date) -> Adjustment:
    def d(x: Any) -> dt.date | None:
        if not x:
            return None
        if isinstance(x, dt.date):
            return x
        return dt.date.fromisoformat(str(x)[:10])

    typ = a.get("type", "one_time")
    date = d(a.get("date"))
    if typ in ("one_time", "recurring") and date is None:
        date = as_of + dt.timedelta(days=14)
    pct = a.get("pct")
    if pct is None and a.get("percent") is not None:
        pct = a.get("percent")
    return Adjustment(
        type=typ,
        amount=float(a.get("amount") or 0),
        kind=a.get("kind", "expense"),
        date=date,
        end_date=d(a.get("end_date")),
        cadence=a.get("cadence", "monthly"),
        category=(resolve_categories(a.get("category")) or [a.get("category")])[0]
        if a.get("category")
        else None,
        pct=float(pct) if pct is not None else None,
        merchants=list(a.get("merchants") or []),
        label=a.get("label", ""),
    )


_ADJ_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["one_time", "recurring", "category_change", "cancel_recurring", "income_change"],
        },
        "amount": {"type": "number", "description": "Positive amount (for one_time/recurring)"},
        "kind": {"type": "string", "enum": ["expense", "income"]},
        "date": {"type": "string", "description": "ISO date (YYYY-MM-DD) of the purchase/start"},
        "end_date": {"type": "string"},
        "cadence": {"type": "string", "enum": ["weekly", "monthly", "quarterly", "yearly"]},
        "category": {"type": "string", "description": "Category id for category_change, e.g. food_delivery"},
        "pct": {"type": "number", "description": "Percent change, e.g. -30 for a 30% cut"},
        "merchants": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Subscription merchants to cancel",
        },
        "label": {"type": "string"},
    },
    "required": ["type"],
}


def _tool(
    name: str, description: str, properties: dict[str, Any] | None = None, required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
        },
    }


TOOL_SPECS: list[dict[str, Any]] = [
    _tool(
        "rank_spending",
        "Rank largest individual expenses, categories or merchants. Use transaction for 'biggest expense', category for 'biggest category', merchant for 'where do I spend most'. Transfers/investments excluded.",
        {
            "group_by": {"type": "string", "enum": ["transaction", "category", "merchant"]},
            "period": {"type": "string"},
            "category": {"type": "string"},
            "merchant": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "limit": {"type": "integer"},
        },
    ),
    _tool(
        "get_overview",
        "Balance, last-30-day income/spend/savings rate, top categories, subscriptions, goals.",
    ),
    _tool(
        "get_spending",
        "Total spend for a category and/or merchant over a period.",
        {
            "category": {
                "type": "string",
                "description": "Category id or plain word, e.g. food_delivery, dining, rent",
            },
            "merchant": {"type": "string"},
            "period": {
                "type": "string",
                "description": "all_time | this_month | last_month | last_30_days | last_90_days | this_year | YYYY-MM",
            },
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
        },
    ),
    _tool(
        "search_transactions",
        "Find individual transactions.",
        {
            "query": {"type": "string"},
            "category": {"type": "string"},
            "min_amount": {"type": "number"},
            "limit": {"type": "integer"},
            "period": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
        },
    ),
    _tool("get_subscriptions", "Active subscriptions with monthly cost, usage signals and price changes."),
    _tool(
        "get_forecast",
        "Projected balance (P10/P50/P90) at a future date, low point, goal odds, upcoming bills.",
        {"days_ahead": {"type": "integer"}, "date": {"type": "string", "description": "YYYY-MM-DD"}},
    ),
    _tool(
        "simulate",
        "What-if simulation against the user's real trajectory. Use for 'can I afford…' questions.",
        {"adjustments": {"type": "array", "items": _ADJ_SCHEMA}},
        ["adjustments"],
    ),
    _tool("get_goals", "Savings goals with probability of success, shortfall and ETA."),
    _tool("get_recommendations", "Current personalised recommendations with their triggers."),
    _tool("get_insights", "Detected patterns and anomalies from the latest analysis."),
    _tool("estimate_trip_cost", "Estimate a trip's cost from the user's own past travel spending."),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}
