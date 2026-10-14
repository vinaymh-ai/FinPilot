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
from finpilot.decisions.jev import get_jev
from finpilot.domain.stats import month_key
from finpilot.domain.taxonomy import as_dicts, is_spending_flow, label
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
        "group": "pipeline",
        "role": "LangGraph state machine: plans the run with a reason for every step, routes work between "
        "specialists, stops dependent work on critical failures and records every step",
    },
    {
        "id": "categorizer",
        "name": "Categorizer",
        "group": "pipeline",
        "role": "Memory → lexicon → keyword phrases → person-to-person detection → one LLM decision per uncertain "
        "merchant → merchant consistency. Every label keeps its source, confidence and reason; debit/credit "
        "direction and payment channel are recorded for each transaction",
    },
    {
        "id": "analyzer",
        "name": "Analyzer",
        "group": "pipeline",
        "role": "Rolling-window trends with the merchants driving each spike, month-to-date pace, robust "
        "anomaly detection, subscription inventory & usage, impulse buys, review queue, behaviour profile",
    },
    {
        "id": "forecaster",
        "name": "Forecaster",
        "group": "pipeline",
        "role": "Monte-Carlo cash-flow projection, goal probabilities (spend vs reserve goals) and what-if "
        "simulation on common random numbers",
    },
    {
        "id": "advisor",
        "name": "Advisor",
        "group": "pipeline",
        "role": "Reasons over patterns, obligations, forecasts and goals to prioritise evidenced actions; "
        "adapts to your feedback and goal progress",
    },
    {
        "id": "auditor",
        "name": "Auditor",
        "group": "pipeline",
        "model_agent": None,
        "role": "Independently verifies every recommendation: trigger matches a finding of this run, evidence "
        "transactions exist, figures are grounded, reasoning is complete; reports categorization health",
    },
    {
        "id": "digest",
        "name": "Digest",
        "group": "pipeline",
        "role": "Scheduled weekly digest (APScheduler): one key insight + one action, by email / Telegram / in-app",
    },
    {
        "id": "chat_planner",
        "name": "Chat planner",
        "group": "chat",
        "model_agent": "chat",
        "role": "Guardrails, then routes each question to the right specialist tools (spending, forecast, "
        "simulation, goals, advice) or the SQL Analyst; resolves follow-ups from history",
    },
    {
        "id": "retriever",
        "name": "Retriever (RAG)",
        "group": "chat",
        "model_agent": None,
        "model_label": "local BM25 + fuzzy",
        "role": "Links your words to merchants and categories that exist in your data (typos included) and "
        "retrieves relevant insights, recommendations and goals",
    },
    {
        "id": "sql_analyst",
        "name": "SQL Analyst",
        "group": "chat",
        "model_agent": "chat",
        "role": "Answers any transaction question — counts, lists, averages, breakdowns, credits, comparisons — "
        "over a sandboxed per-user SQL view; deterministic when certain, LLM text-to-SQL with self-repair otherwise",
    },
    {
        "id": "chat",
        "name": "Composer & verifier",
        "group": "chat",
        "role": "Writes the answer from verified results; numeric grounding and an independent claim check "
        "reject anything the evidence does not support",
    },
]


@public.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    db.execute(text("SELECT 1"))
    r = get_router()
    return {
        "status": "ok",
        "version": __version__,
        "llm": {"enabled": r.enabled, "advisor": r.describe("advisor")},
        "decisions": get_jev().describe(),
        "scheduler": scheduler.running,
        "time": dt.datetime.now(dt.UTC).isoformat(),
    }


@router.get("/meta")
def meta(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    user = repo.get_or_create_user(db, uid)
    repo.custom_categories(db, uid)
    r = get_router()
    cards = []
    for c in AGENT_CARDS:
        agent = c.get("model_agent", c["id"])
        model = r.describe(agent) if agent in AGENTS and r.available(agent) else "rules"
        if c["id"] == "coordinator":
            model = "langgraph"
        elif c.get("model_label"):
            model = c["model_label"]
        cards.append(
            {k: v for k, v in c.items() if k not in ("model_agent", "model_label")} | {"model": model}
        )
    return {
        "version": __version__,
        "currency": user.currency,
        "user": {"id": user.id, "name": user.name},
        "categories": as_dicts(uid),
        "agents": cards,
        "llm_enabled": r.enabled,
        "decisions": get_jev().describe(),
    }


@router.get("/graph")
def graph() -> dict[str, Any]:
    return {"mermaid": Coordinator.mermaid()}


def _analysis_running(db: Session, uid: int) -> bool:
    """A pipeline run in progress (ignoring ones stuck for over 15 minutes, e.g. after a crash)."""
    from finpilot.models import AgentRun

    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=15)
    return (
        db.scalar(
            select(AgentRun.id)
            .where(
                AgentRun.user_id == uid,
                AgentRun.graph == "pipeline",
                AgentRun.status == "running",
                AgentRun.started_at >= cutoff,
            )
            .limit(1)
        )
        is not None
    )


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
        elif is_spending_flow(t.category, t.amount):
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
        "analysis_running": _analysis_running(db, uid),
        "latest_digest": ser.digest(latest_digest) if latest_digest else None,
        "goals": [
            ser.goal(g)
            for g in db.scalars(select(Goal).where(repo.goal_scope(db, uid), Goal.status == "active"))
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
