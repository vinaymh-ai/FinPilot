"""Data access helpers bridging ORM rows and the agents' plain domain objects."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from finpilot.agents.forecaster import GoalInput
from finpilot.config import get_settings
from finpilot.domain.recurring import RecurringCandidate, TxnLite
from finpilot.models import (
    AgentRun,
    AppSetting,
    Forecast,
    Goal,
    Insight,
    MerchantMemory,
    MerchantProfile,
    Recommendation,
    RecurringSeries,
    Transaction,
    User,
)

DEFAULT_USER_ID = 1


def get_or_create_user(session: Session, user_id: int = DEFAULT_USER_ID) -> User:
    user = session.get(User, user_id)
    if user is None:
        s = get_settings()
        user = User(id=user_id, name="Demo User", currency=s.currency, current_balance=0.0)
        session.add(user)
        session.commit()
    return user


def load_transactions(session: Session, user_id: int) -> list[Transaction]:
    return list(
        session.scalars(
            select(Transaction)
            .where(Transaction.user_id == user_id)
            .order_by(Transaction.date, Transaction.time, Transaction.id)
        )
    )


def to_lite(t: Transaction) -> TxnLite:
    return TxnLite(
        id=t.id,
        date=t.date,
        amount=float(t.amount),
        merchant=t.merchant,
        category=t.category or "other",
        description=t.description,
        time=t.time,
    )


def current_balance(session: Session, user: User) -> tuple[float, dt.date]:
    """Balance as of the latest transaction: anchor balance + flows recorded after the anchor date."""
    last = session.scalar(select(func.max(Transaction.date)).where(Transaction.user_id == user.id))
    as_of = last or dt.date.today()
    bal = float(user.current_balance or 0.0)
    if user.balance_as_of is not None:
        after = session.scalar(
            select(func.coalesce(func.sum(Transaction.amount), 0.0)).where(
                Transaction.user_id == user.id, Transaction.date > user.balance_as_of
            )
        )
        bal += float(after or 0.0)
    return round(bal, 2), max(as_of, user.balance_as_of or as_of)


def recurring_candidates(session: Session, user_id: int) -> list[RecurringCandidate]:
    rows = list(session.scalars(select(RecurringSeries).where(RecurringSeries.user_id == user_id)))
    ids: dict[int, list[int]] = defaultdict(list)
    for tid, rid in session.execute(
        select(Transaction.id, Transaction.recurring_id).where(
            Transaction.user_id == user_id, Transaction.recurring_id.is_not(None)
        )
    ):
        ids[rid].append(tid)
    from finpilot.domain.lexicon import rule_by_name

    families = merchant_families(session, user_id)
    out = []
    for r in rows:
        rule = rule_by_name(r.merchant)
        out.append(
            RecurringCandidate(
                merchant=r.merchant,
                category=r.category,
                kind=r.kind,
                cadence=r.cadence,
                interval_days=r.interval_days,
                typical_amount=float(r.typical_amount),
                last_amount=float(r.last_amount),
                first_date=r.first_date,
                last_date=r.last_date,
                next_expected=r.next_expected,
                occurrences=r.occurrences,
                regularity=r.regularity,
                price_change_pct=r.price_change_pct,
                status="cancelled" if r.status == "cancelled" else r.status,
                txn_ids=ids.get(r.id, []),
                # lexicon knows the big brands; Jev's service family covers everything else
                group=(rule.group if rule else None) or families.get(r.merchant.upper()),
                parent=rule.parent if rule else None,
            )
        )
    return out


def save_recurring(session: Session, user_id: int, candidates: list[RecurringCandidate]) -> dict[str, str]:
    """Upsert detected series, preserving user-set flags (usage, cancelled). Returns usage overrides."""
    existing = list(session.scalars(select(RecurringSeries).where(RecurringSeries.user_id == user_id)))
    by_key = {(r.merchant, r.cadence, r.typical_amount > 0): r for r in existing}
    seen: set[int] = set()
    session.execute(
        update(Transaction)
        .where(Transaction.user_id == user_id)
        .values(is_recurring=False, recurring_id=None)
    )
    for c in candidates:
        row = by_key.get((c.merchant, c.cadence, c.typical_amount > 0))
        fields = dict(
            category=c.category,
            kind=c.kind,
            interval_days=c.interval_days,
            typical_amount=c.typical_amount,
            last_amount=c.last_amount,
            first_date=c.first_date,
            last_date=c.last_date,
            next_expected=c.next_expected,
            occurrences=c.occurrences,
            regularity=c.regularity,
            price_change_pct=c.price_change_pct,
        )
        if row is None:
            row = RecurringSeries(
                user_id=user_id, merchant=c.merchant, cadence=c.cadence, status=c.status, **fields
            )
            session.add(row)
            session.flush()
        else:
            for k, v in fields.items():
                setattr(row, k, v)
            if row.status == "cancelled":
                # charged again after the user cancelled → the cancellation did not go through
                if row.cancelled_on and c.last_date > row.cancelled_on:
                    row.status, row.cancelled_on = c.status, None
                    row.usage_signal = f"Charged again on {c.last_date:%d %b} after you marked it cancelled"
            else:
                row.status = c.status
        seen.add(row.id)
        if c.txn_ids:
            session.execute(
                update(Transaction)
                .where(Transaction.id.in_(c.txn_ids))
                .values(is_recurring=True, recurring_id=row.id)
            )
    for r in existing:
        if r.id not in seen and r.status != "cancelled":
            r.status = "lapsed"
    session.flush()
    return {
        r.merchant: r.usage
        for r in session.scalars(select(RecurringSeries).where(RecurringSeries.user_id == user_id))
        if r.usage != "unknown"
    }


def goal_inputs(session: Session, user_id: int) -> list[GoalInput]:
    return [
        GoalInput(
            g.id,
            g.name,
            float(g.target_amount),
            float(g.saved_amount),
            g.deadline,
            g.priority,
            spend_at_deadline=bool(g.spend_at_deadline),
        )
        for g in session.scalars(select(Goal).where(Goal.user_id == user_id, Goal.status == "active"))
    ]


def goal_history(session: Session, user_id: int) -> dict[str, list[dict[str, Any]]]:
    return {
        g.name: list(g.history or []) for g in session.scalars(select(Goal).where(Goal.user_id == user_id))
    }


def append_goal_snapshots(
    session: Session, user_id: int, forecast_goals: list[dict[str, Any]], run_id: int
) -> None:
    by_id = {g["id"]: g for g in forecast_goals}
    for goal in session.scalars(select(Goal).where(Goal.user_id == user_id)):
        fg = by_id.get(goal.id)
        if not fg:
            continue
        hist = list(goal.history or [])
        hist.append(
            {
                "date": dt.date.today().isoformat(),
                "run_id": run_id,
                "probability": fg["probability"],
                "shortfall": fg["shortfall"],
                "eta": fg.get("eta"),
                "saved": float(goal.saved_amount),
            }
        )
        goal.history = hist[-52:]
        if fg["status"] == "achieved" and goal.status == "active":
            goal.status = "achieved"


def merchant_memory(session: Session, user_id: int) -> dict[str, str]:
    return {
        m.merchant_key: m.category
        for m in session.scalars(select(MerchantMemory).where(MerchantMemory.user_id == user_id))
    }


def merchant_history(session: Session, user_id: int) -> dict[str, str]:
    """Merchants previously categorised by a model or the user — consistency, and never paying twice."""
    out: dict[str, str] = {}
    for merchant, category in session.execute(
        select(Transaction.merchant, Transaction.category).where(
            Transaction.user_id == user_id, Transaction.category_source.in_(("llm", "jev", "user"))
        )
    ):
        if category:
            out[merchant.upper()] = category
    return out


NON_OVERLAP_FAMILIES = {"none", "utility_bill"}


def merchant_families(session: Session, user_id: int) -> dict[str, str]:
    """Service family per merchant (from System One profiles) — used to spot overlapping subscriptions."""
    return {
        p.merchant_key: p.service_family
        for p in session.scalars(
            select(MerchantProfile).where(
                MerchantProfile.user_id == user_id, MerchantProfile.service_family.is_not(None)
            )
        )
        if p.service_family not in NON_OVERLAP_FAMILIES
    }


def save_merchant_profiles(session: Session, user_id: int, profiles: list[Any]) -> int:
    """Upsert what Jev decided about each merchant (category, payee type, service family)."""
    if not profiles:
        return 0
    existing = {
        p.merchant_key: p
        for p in session.scalars(select(MerchantProfile).where(MerchantProfile.user_id == user_id))
    }
    for d in profiles:
        key = d.merchant.upper()
        row = existing.get(key)
        fields = dict(
            category=d.category,
            confidence=d.confidence,
            payee_type=d.payee_type,
            service_family=d.service_family,
            probabilities=d.probabilities,
            source="jev",
        )
        if row is None:
            row = MerchantProfile(user_id=user_id, merchant_key=key, **fields)
            session.add(row)
            existing[key] = row
        else:
            for k, v in fields.items():
                setattr(row, k, v)
    return len(profiles)


def remember_correction(session: Session, user_id: int, merchant: str, category: str) -> int:
    """Store a user correction and apply it to all transactions from the same merchant."""
    key = merchant.upper()
    row = session.scalar(
        select(MerchantMemory).where(MerchantMemory.user_id == user_id, MerchantMemory.merchant_key == key)
    )
    if row is None:
        session.add(MerchantMemory(user_id=user_id, merchant_key=key, category=category, hits=1))
    else:
        row.category, row.hits = category, row.hits + 1
    res = session.execute(
        update(Transaction)
        .where(Transaction.user_id == user_id, Transaction.merchant == merchant)
        .values(
            category=category,
            category_source="user",
            category_confidence=1.0,
            category_reason="Corrected by you",
        )
    )
    return res.rowcount or 0


def latest_run(session: Session, user_id: int, graph: str = "pipeline") -> AgentRun | None:
    return session.scalar(
        select(AgentRun)
        .where(AgentRun.user_id == user_id, AgentRun.graph == graph, AgentRun.status == "succeeded")
        .order_by(AgentRun.id.desc())
        .limit(1)
    )


def latest_forecast(session: Session, user_id: int) -> Forecast | None:
    return session.scalar(
        select(Forecast)
        .where(Forecast.user_id == user_id, Forecast.run_id.is_not(None))
        .order_by(Forecast.id.desc())
        .limit(1)
    )


def latest_recommendations(session: Session, user_id: int) -> list[Recommendation]:
    run = latest_run(session, user_id)
    if run is None:
        return []
    return list(
        session.scalars(
            select(Recommendation)
            .where(Recommendation.run_id == run.id)
            .order_by(Recommendation.priority.desc())
        )
    )


def latest_insights(session: Session, user_id: int) -> list[Insight]:
    run = latest_run(session, user_id)
    if run is None:
        return []
    return list(session.scalars(select(Insight).where(Insight.run_id == run.id).order_by(Insight.id)))


def recommendation_feedback(session: Session, user_id: int) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(Recommendation)
        .where(Recommendation.user_id == user_id, Recommendation.status != "new")
        .order_by(Recommendation.id)
    )
    return [
        {
            "fingerprint": r.fingerprint,
            "kind": r.kind,
            "status": r.status,
            "monthly_impact": r.monthly_impact,
            "updated_at": (r.updated_at or r.created_at),
        }
        for r in rows
    ]


def get_setting(session: Session, key: str, default: Any = None) -> Any:
    row = session.get(AppSetting, key)
    return row.value if row else default


def put_setting(session: Session, key: str, value: Any) -> None:
    row = session.get(AppSetting, key)
    if row is None:
        session.add(AppSetting(key=key, value=value))
    else:
        row.value = value


def wipe_user_data(session: Session, user_id: int) -> None:
    for model in (
        Recommendation,
        Insight,
        Forecast,
        AgentRun,
        Transaction,
        RecurringSeries,
        MerchantMemory,
        MerchantProfile,
        Goal,
    ):
        session.execute(delete(model).where(model.user_id == user_id))
