"""Health, metadata, dashboard, agent graph and evaluation results."""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from finpilot import __version__
from finpilot.agents.coordinator import Coordinator
from finpilot.api import serializers as ser
from finpilot.api.deps import current_user_id, get_db
from finpilot.config import ROOT_DIR, get_settings
from finpilot.domain.stats import month_key
from finpilot.domain.taxonomy import NON_SPEND, as_dicts, label
from finpilot.llm.client import AGENTS, get_router
from finpilot.models import Digest, Goal
from finpilot.services import repository as repo
from finpilot.services.scheduler import scheduler
from finpilot.services.toolkit import FinanceToolkit

router = APIRouter()
public = APIRouter()

AGENT_CARDS = [
    {
        "id": "coordinator",
        "name": "Coordinator",
        "role": "LangGraph state machine: plans the run and routes work "
        "between specialists; records every step",
    },
    {
        "id": "categorizer",
        "name": "Categorizer",
        "role": "Memory → merchant lexicon → keyword evidence → batched LLM "
        "for the long tail → recurring-pattern post-pass",
    },
    {
        "id": "analyzer",
        "name": "Analyzer",
        "role": "Rolling-window trends, robust anomaly detection, subscription "
        "inventory & usage, impulse buys, behaviour profile",
    },
    {
        "id": "forecaster",
        "name": "Forecaster",
        "role": "Monte-Carlo cash-flow projection, goal probabilities and "
        "what-if simulation on common random numbers",
    },
    {
        "id": "advisor",
        "name": "Advisor",
        "role": "Ranks specific, quantified actions with trigger + evidence; "
        "LLM phrasing behind a numeric grounding check",
    },
    {
        "id": "chat",
        "name": "Chat",
        "role": "Tool-calling assistant over your real data (simulate, spending, goals…)",
    },
    {"id": "digest", "name": "Digest", "role": "Scheduled weekly digest: one key insight + one action"},
]


@public.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    db.execute(text("SELECT 1"))
    r = get_router()
    return {
        "status": "ok",
        "version": __version__,
        "llm": {"enabled": r.enabled, "advisor": r.describe("advisor")},
        "scheduler": scheduler.running,
        "time": dt.datetime.now(dt.UTC).isoformat(),
    }


@router.get("/meta")
def meta(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    user = repo.get_or_create_user(db, uid)
    r = get_router()
    cards = []
    for c in AGENT_CARDS:
        model = r.describe(c["id"]) if c["id"] in AGENTS and r.available(c["id"]) else "rules"
        cards.append({**c, "model": model if c["id"] != "coordinator" else "langgraph"})
    return {
        "version": __version__,
        "currency": user.currency,
        "user": {"id": user.id, "name": user.name},
        "categories": as_dicts(),
        "agents": cards,
        "llm_enabled": r.enabled,
    }


@router.get("/graph")
def graph() -> dict[str, Any]:
    return {"mermaid": Coordinator.mermaid()}


@router.get("/dashboard")
def dashboard(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    kit = FinanceToolkit(db, uid)
    overview = kit.get_overview()
    run = repo.latest_run(db, uid)
    fc = repo.latest_forecast(db, uid)
    recs = repo.latest_recommendations(db, uid)
    insights = [i for i in repo.latest_insights(db, uid) if i.kind != "subscription_inventory"]
    # monthly cash-flow from transactions (last 8 months)
    months: dict[str, dict[str, float]] = defaultdict(lambda: {"income": 0.0, "spend": 0.0, "invested": 0.0})
    cats_30: dict[str, float] = defaultdict(float)
    start30 = kit.as_of - dt.timedelta(days=29)
    for t in kit.txns:
        m = months[month_key(t.date)]
        if t.category == "income":
            m["income"] += t.amount
        elif t.category == "investments":
            m["invested"] += -t.amount
        elif t.category not in NON_SPEND:
            m["spend"] += -t.amount
            if t.date >= start30:
                cats_30[t.category] += -t.amount
    cashflow = [
        {
            "month": k,
            **{kk: round(vv, 2) for kk, vv in v.items()},
            "net": round(v["income"] - v["spend"] - v["invested"], 2),
        }
        for k, v in sorted(months.items())
    ][-8:]
    latest_digest = db.scalar(select(Digest).where(Digest.user_id == uid).order_by(Digest.id.desc()).limit(1))
    analysis = run.analysis if run is not None else None
    return {
        "overview": overview,
        "cashflow": cashflow,
        "categories_30d": [
            {"category": c, "label": label(c), "amount": round(v, 2)}
            for c, v in sorted(cats_30.items(), key=lambda kv: -kv[1])
        ],
        "forecast": fc.payload if fc else None,
        "insights": [ser.insight(i) for i in insights[:6]],
        "recommendations": [ser.recommendation(r) for r in recs[:4]],
        "headline": (run.summary or {}).get("headline") if run else None,
        "profile": (analysis or {}).get("profile", []),
        "last_run": ser.run(run) if run else None,
        "latest_digest": ser.digest(latest_digest) if latest_digest else None,
        "goals": [
            ser.goal(g) for g in db.scalars(select(Goal).where(Goal.user_id == uid, Goal.status == "active"))
        ],
    }


@router.get("/evaluation")
def evaluation() -> dict[str, Any]:
    path = ROOT_DIR / "evaluation" / "results" / "latest.json"
    if not path.exists():
        return {"available": False, "hint": "run `python -m evaluation.run_eval` to generate results"}
    data = json.loads(path.read_text())
    data["available"] = True
    return data


@router.get("/config/public")
def public_config() -> dict[str, Any]:
    s = get_settings()
    return {
        "currency": s.currency,
        "timezone": s.timezone,
        "forecast_horizon_days": s.forecast_horizon_days,
        "monte_carlo_paths": s.monte_carlo_paths,
    }
