"""System One (Jev): decision log, cost/savings stats, runtime settings and a live connection test."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from finpilot.api.deps import current_user_id, get_db, require_admin
from finpilot.config import get_settings
from finpilot.decisions.costs import savings_report
from finpilot.decisions.jev import Choice, JevClient, JevError, get_jev
from finpilot.models import Decision, LLMCall
from finpilot.schemas import JevSettingsUpdate
from finpilot.services import repository as repo

router = APIRouter()


def decision_row(d: Decision) -> dict[str, Any]:
    return {
        "id": d.id,
        "run_id": d.run_id,
        "agent": d.agent,
        "purpose": d.purpose,
        "subject": d.subject,
        "question": d.question,
        "answer": d.answer,
        "confidence": d.confidence,
        "probabilities": d.probabilities or {},
        "action": d.action,
        "model": d.model,
        "latency_ms": d.latency_ms,
        "cached": d.cached,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }


@router.get("/decisions")
def list_decisions(
    purpose: str | None = None,
    action: str | None = None,
    run_id: int | None = None,
    limit: int = 100,
    uid: int = Depends(current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    q = (
        select(Decision)
        .where(Decision.user_id == uid)
        .order_by(Decision.id.desc())
        .limit(min(max(limit, 1), 500))
    )
    if purpose:
        q = q.where(Decision.purpose == purpose)
    if action:
        q = q.where(Decision.action == action)
    if run_id is not None:
        q = q.where(Decision.run_id == run_id)
    return {"decisions": [decision_row(d) for d in db.scalars(q)]}


@router.get("/decisions/stats")
def decision_stats(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    kind_expression = func.coalesce(LLMCall.kind, "llm")
    for kind, n, tin, tout, cost, cached, ok, lat in db.execute(
        select(
            kind_expression,
            func.count(),
            func.sum(LLMCall.prompt_tokens),
            func.sum(LLMCall.completion_tokens),
            func.sum(LLMCall.cost_usd),
            func.sum(case((LLMCall.cached.is_(True), 1), else_=0)),
            func.sum(case((LLMCall.success.is_(True), 1), else_=0)),
            func.avg(LLMCall.latency_ms),
        )
        .where(LLMCall.user_id == uid)
        .group_by(kind_expression)
    ):
        by_kind[kind] = {
            "calls": n,
            "input_tokens": int(tin or 0),
            "output_tokens": int(tout or 0),
            "cost_usd": round(float(cost or 0), 6),
            "cache_hits": int(cached or 0),
            "successes": int(ok or 0),
            "avg_latency_ms": round(float(lat or 0)),
        }
    jev_by_purpose: dict[str, dict[str, Any]] = {}
    for purpose, n, tin, cost in db.execute(
        select(LLMCall.purpose, func.count(), func.sum(LLMCall.prompt_tokens), func.sum(LLMCall.cost_usd))
        .where(LLMCall.kind == "jev", LLMCall.user_id == uid)
        .group_by(LLMCall.purpose)
    ):
        jev_by_purpose[purpose or "?"] = {
            "calls": n,
            "input_tokens": int(tin or 0),
            "cost_usd": float(cost or 0),
        }
    acts: dict[str, dict[str, int]] = {}
    conf: dict[str, float] = {}
    for purpose, action, n, avg_conf in db.execute(
        select(Decision.purpose, Decision.action, func.count(), func.avg(Decision.confidence))
        .where(Decision.user_id == uid)
        .group_by(Decision.purpose, Decision.action)
    ):
        acts.setdefault(purpose, {})[action] = n
        conf[f"{purpose}:{action}"] = round(float(avg_conf or 0), 3)
    return {
        "by_kind": by_kind,
        "decisions": {
            p: {"actions": a, "total": sum(a.values()), "avg_confidence": {k: conf[f"{p}:{k}"] for k in a}}
            for p, a in acts.items()
        },
        "savings": savings_report(jev_by_purpose, acts),
        "decision_model": get_jev().describe(),
    }


# ---------------------------------------------------------------------------- settings


def jev_state() -> dict[str, Any]:
    j = get_jev()
    s = get_settings()
    url = (
        f"{s.typesafe_base_url.rstrip('/')}/systemone"
        if j.provider == "typesafe"
        else f"{s.mesh_base_url.rstrip('/')}/evaluate"
    )
    return {
        "enabled": j.enabled,
        "configured": j.configured,
        "available": j.available(),
        "provider": j.provider,
        "model": j.model,
        "endpoint": url,
        "key_env": "TYPESAFE_API_KEY" if j.provider == "typesafe" else "MESH_API_KEY",
        "overrides": j.overrides,
        "accept_confidence": j.accept_confidence,
        "route_confidence": j.route_confidence,
        "guard_confidence": j.guard_confidence,
        "escalate_to_llm": j.escalate_to_llm,
        "order_check": j.order_check,
        "price_in_per_1m": s.jev_price_in_per_1m,
        "used_for": [
            {
                "purpose": "merchant_category",
                "agent": "categorizer",
                "replaces": "batched LLM classification",
            },
            {"purpose": "service_family", "agent": "categorizer", "replaces": "lexicon-only overlap groups"},
            {"purpose": "chat_route", "agent": "chat", "replaces": "LLM tool-calling for simple questions"},
            {"purpose": "faithfulness", "agent": "advisor/analyzer/forecaster/digest/chat", "replaces": "—"},
            {"purpose": "goal_type", "agent": "forecaster", "replaces": "—"},
            {"purpose": "csv_column", "agent": "ingestion", "replaces": "—"},
        ],
    }


@router.get("/settings/decisions")
def get_decision_settings() -> dict[str, Any]:
    return jev_state()


@router.put("/settings/decisions", dependencies=[Depends(require_admin)])
def update_decision_settings(body: JevSettingsUpdate, db: Session = Depends(get_db)) -> dict[str, Any]:
    j = get_jev()
    ov = j.overrides
    ov.update(body.model_dump(exclude_unset=True))
    ov = {k: v for k, v in ov.items() if v is not None}
    j.set_overrides(ov)
    repo.put_setting(db, "jev_overrides", ov)
    db.commit()
    return jev_state()


@router.post("/settings/decisions/test", dependencies=[Depends(require_admin)])
def test_decisions() -> dict[str, Any]:
    """One tiny live evaluation (bypassing the cache) to prove the Jev route works end to end."""
    base = get_jev()
    probe = JevClient(base.settings)
    probe.set_overrides({"enabled": True})
    if not probe.configured:
        return {"ok": False, "provider": probe.provider, "error": f"{jev_state()['key_env']} is not set"}
    t0 = time.perf_counter()
    try:
        ev = probe.evaluate(
            {"statement_text": "UPI-STARBUCKS COFFEE-INDIRANAGAR", "amount": "small (under ₹500)"},
            {
                "category": Choice(
                    "Which spending category is this card transaction?",
                    {
                        "dining": "Cafés, restaurants, bars",
                        "groceries": "Supermarkets and grocery apps",
                        "other": "Anything else",
                    },
                )
            },
            agent="settings",
            purpose="connection_test",
            cache=False,
        )
    except JevError as exc:
        return {"ok": False, "provider": probe.provider, "error": str(exc)[:400]}
    a = ev["category"]
    return {
        "ok": True,
        "provider": probe.provider,
        "model": ev.model,
        "answer": a.choice,
        "confidence": round(a.confidence, 3),
        "probabilities": a.probabilities,
        "input_tokens": ev.input_tokens,
        "cost_usd": ev.cost_usd,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }
