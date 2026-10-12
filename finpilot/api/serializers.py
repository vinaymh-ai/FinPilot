"""ORM → JSON helpers for API responses."""

from __future__ import annotations

from typing import Any

from finpilot.domain.recurring import monthly_equivalent
from finpilot.domain.taxonomy import label
from finpilot.models import (
    AgentRun,
    AgentStep,
    Digest,
    Goal,
    Insight,
    Recommendation,
    RecurringSeries,
    Transaction,
)


def txn(t: Transaction) -> dict[str, Any]:
    return {
        "id": t.id,
        "date": t.date.isoformat(),
        "time": t.time.strftime("%H:%M") if t.time else None,
        "description": t.description,
        "merchant": t.merchant,
        "amount": float(t.amount),
        "category": t.category,
        "category_label": label(t.category),
        "category_source": t.category_source,
        "category_confidence": t.category_confidence,
        "category_reason": t.category_reason,
        "tags": t.tags or [],
        "is_recurring": t.is_recurring,
        "is_anomaly": t.is_anomaly,
        "anomaly_reason": t.anomaly_reason,
        "notes": t.notes,
    }


def recurring(r: RecurringSeries) -> dict[str, Any]:
    amt = abs(float(r.typical_amount))
    return {
        "id": r.id,
        "merchant": r.merchant,
        "category": r.category,
        "category_label": label(r.category),
        "kind": r.kind,
        "cadence": r.cadence,
        "amount": amt,
        "signed_amount": float(r.typical_amount),
        "monthly_cost": round(monthly_equivalent(amt, r.cadence), 2),
        "last_amount": abs(float(r.last_amount)),
        "first_date": r.first_date.isoformat(),
        "last_date": r.last_date.isoformat(),
        "next_expected": r.next_expected.isoformat(),
        "occurrences": r.occurrences,
        "regularity": r.regularity,
        "price_change_pct": r.price_change_pct,
        "status": r.status,
        "usage": r.usage,
        "usage_signal": r.usage_signal,
        "cancelled_on": r.cancelled_on.isoformat() if r.cancelled_on else None,
    }


def insight(i: Insight) -> dict[str, Any]:
    return {
        "id": i.id,
        "run_id": i.run_id,
        "kind": i.kind,
        "severity": i.severity,
        "title": i.title,
        "detail": i.detail,
        "metrics": i.metrics or {},
        "transaction_ids": i.transaction_ids or [],
        "created_at": i.created_at.isoformat() if i.created_at else None,
    }


def recommendation(r: Recommendation) -> dict[str, Any]:
    return {
        "id": r.id,
        "run_id": r.run_id,
        "kind": r.kind,
        "title": r.title,
        "message": r.message,
        "action": r.action,
        "monthly_impact": float(r.monthly_impact),
        "priority": r.priority,
        "urgency": r.urgency,
        "status": r.status,
        "trigger": r.trigger,
        "evidence": r.evidence,
        "reasoning": r.reasoning,
        "generated_by": r.generated_by,
        "grounded": r.grounded,
        "fingerprint": r.fingerprint,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def goal(g: Goal, projection: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": g.id,
        "name": g.name,
        "target_amount": float(g.target_amount),
        "saved_amount": float(g.saved_amount),
        "deadline": g.deadline.isoformat(),
        "priority": g.priority,
        "status": g.status,
        "progress_pct": round(float(g.saved_amount) / float(g.target_amount) * 100, 1)
        if g.target_amount
        else 0,
        "history": g.history or [],
        "projection": projection,
        "goal_type": g.goal_type,
        "spend_at_deadline": g.spend_at_deadline,
        "type_source": g.type_source,
        "type_confidence": g.type_confidence,
    }


def step(s: AgentStep) -> dict[str, Any]:
    return {
        "id": s.id,
        "agent": s.agent,
        "status": s.status,
        "started_at": s.started_at.isoformat(),
        "duration_ms": s.duration_ms,
        "summary": s.summary,
        "details": s.details or {},
        "llm_calls": s.llm_calls,
        "error": s.error,
    }


def run(r: AgentRun, with_steps: bool = False) -> dict[str, Any]:
    out = {
        "id": r.id,
        "trigger": r.trigger,
        "graph": r.graph,
        "status": r.status,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "duration_ms": int((r.finished_at - r.started_at).total_seconds() * 1000)
        if r.finished_at and r.started_at
        else None,
        "summary": r.summary or {},
        "error": r.error,
    }
    if with_steps:
        out["steps"] = [step(s) for s in r.steps]
    return out


def digest(d: Digest) -> dict[str, Any]:
    return {
        "id": d.id,
        "run_id": d.run_id,
        "trigger": d.trigger,
        "subject": d.subject,
        "headline": d.headline,
        "action": d.action,
        "body_markdown": d.body_markdown,
        "deliveries": d.deliveries or {},
        "recommendation_id": d.recommendation_id,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }
