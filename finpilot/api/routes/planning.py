"""Insights, forecast, what-if simulation, recommendations (decision trail) and goals."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from finpilot.api import serializers as ser
from finpilot.api.deps import current_user_id, get_db
from finpilot.models import Goal, Insight, Recommendation
from finpilot.schemas import GoalCreate, GoalUpdate, RecommendationUpdate, SimulationRequest, dump
from finpilot.services import repository as repo
from finpilot.services.goal_types import GoalTyping, apply_typing, classify_goal, rules_goal_type
from finpilot.services.toolkit import FinanceToolkit

router = APIRouter()


@router.get("/insights")
def insights(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    run = repo.latest_run(db, uid)
    if run is None:
        return {"run": None, "insights": [], "analysis": None}
    rows = list(db.scalars(select(Insight).where(Insight.run_id == run.id).order_by(Insight.id)))
    return {"run": ser.run(run), "insights": [ser.insight(i) for i in rows], "analysis": run.analysis}


@router.get("/forecast")
def forecast(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    fc = repo.latest_forecast(db, uid)
    if fc is None:
        raise HTTPException(404, "no forecast yet — run the agents first")
    return {
        "run_id": fc.run_id,
        "created_at": fc.created_at.isoformat() if fc.created_at else None,
        **fc.payload,
    }


@router.post("/simulate")
def simulate(
    body: SimulationRequest, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    kit = FinanceToolkit(db, uid)
    if not kit.txns:
        raise HTTPException(409, "no transactions yet")
    adjustments = [a.model_dump() for a in body.adjustments]
    for a in adjustments:
        a["date"] = a["date"].isoformat() if a.get("date") else None
        a["end_date"] = a["end_date"].isoformat() if a.get("end_date") else None
    return kit.simulate(adjustments, full=True)


@router.get("/recommendations")
def recommendations(
    include_history: bool = False, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    run = repo.latest_run(db, uid)
    current = repo.latest_recommendations(db, uid)
    out: dict[str, Any] = {
        "run": ser.run(run) if run else None,
        "headline": (run.summary or {}).get("headline") if run else None,
        "recommendations": [ser.recommendation(r) for r in current],
    }
    if include_history:
        hist = db.scalars(
            select(Recommendation)
            .where(Recommendation.user_id == uid, Recommendation.status != "new")
            .order_by(Recommendation.id.desc())
            .limit(50)
        )
        out["history"] = [ser.recommendation(r) for r in hist]
    return out


@router.get("/recommendations/{rec_id}")
def recommendation_detail(
    rec_id: int, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    r = db.get(Recommendation, rec_id)
    if r is None or r.user_id != uid:
        raise HTTPException(404, "recommendation not found")
    from finpilot.models import AgentRun, Transaction

    run = db.get(AgentRun, r.run_id)
    ids = (r.evidence or {}).get("transaction_ids") or []
    txns = list(db.scalars(select(Transaction).where(Transaction.id.in_(ids[:25])))) if ids else []
    earlier = db.scalars(
        select(Recommendation)
        .where(Recommendation.fingerprint == r.fingerprint, Recommendation.id != r.id)
        .order_by(Recommendation.id.desc())
        .limit(10)
    )
    return {
        "recommendation": ser.recommendation(r),
        "run": ser.run(run, with_steps=True) if run else None,
        "transactions": [ser.txn(t) for t in txns],
        "previous": [
            {
                "id": p.id,
                "run_id": p.run_id,
                "status": p.status,
                "monthly_impact": p.monthly_impact,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in earlier
        ],
    }


@router.patch("/recommendations/{rec_id}")
def update_recommendation(
    rec_id: int,
    body: RecommendationUpdate,
    uid: int = Depends(current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    r = db.get(Recommendation, rec_id)
    if r is None or r.user_id != uid:
        raise HTTPException(404, "recommendation not found")
    r.status = body.status
    db.commit()
    db.refresh(r)
    return {"recommendation": ser.recommendation(r)}


# ---------------------------------------------------------------------------- goals
def _projections(db: Session, uid: int) -> dict[int, dict[str, Any]]:
    fc = repo.latest_forecast(db, uid)
    return {g["id"]: g for g in (fc.payload.get("goals", []) if fc else []) if g.get("id") is not None}


@router.get("/goals")
def list_goals(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    proj = _projections(db, uid)
    goals = db.scalars(select(Goal).where(Goal.user_id == uid).order_by(Goal.priority, Goal.deadline))
    return {"goals": [ser.goal(g, proj.get(g.id)) for g in goals]}


@router.post("/goals", status_code=201)
def create_goal(
    body: GoalCreate, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    data = body.model_dump()
    user_type, user_spend = data.pop("goal_type"), data.pop("spend_at_deadline")
    g = Goal(user_id=uid, **data)
    _type_goal(db, uid, g, user_type, user_spend)
    db.add(g)
    db.commit()
    return {"goal": ser.goal(g), "hint": "Run the agents to project this goal"}


def _type_goal(db: Session, uid: int, g: Goal, user_type: str | None, user_spend: bool | None) -> None:
    """A user's explicit choice wins; otherwise System One (or the keyword fallback) decides."""
    if user_type is not None or user_spend is not None:
        guess = rules_goal_type(g.name)
        gtype = user_type or (g.goal_type if g.type_source == "user" else guess.goal_type)
        spend = user_spend if user_spend is not None else guess.spend_at_deadline
        apply_typing(g, GoalTyping(gtype, spend, "user", 1.0))
        return
    if g.type_source == "user":
        return
    user = repo.get_or_create_user(db, uid)
    _, as_of = repo.current_balance(db, user)
    apply_typing(g, classify_goal(g.name, float(g.target_amount), g.deadline, as_of, user.currency))


@router.patch("/goals/{goal_id}")
def update_goal(
    goal_id: int, body: GoalUpdate, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    g = db.get(Goal, goal_id)
    if g is None or g.user_id != uid:
        raise HTTPException(404, "goal not found")
    changes = dump(body)
    user_type, user_spend = changes.pop("goal_type", None), changes.pop("spend_at_deadline", None)
    renamed = "name" in changes and changes["name"] != g.name
    for k, v in changes.items():
        setattr(g, k, v)
    if user_type is not None or user_spend is not None or renamed:
        _type_goal(db, uid, g, user_type, user_spend)
    db.commit()
    return {"goal": ser.goal(g, _projections(db, uid).get(g.id))}


@router.delete("/goals/{goal_id}", status_code=204)
def delete_goal(goal_id: int, uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> None:
    g = db.get(Goal, goal_id)
    if g is None or g.user_id != uid:
        raise HTTPException(404, "goal not found")
    db.delete(g)
    db.commit()
