"""Agent runs (decision trail), chat, LLM usage, digests and the scheduler."""

from __future__ import annotations

import datetime as dt
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from finpilot.agents.chat import ChatAgent
from finpilot.agents.coordinator import Coordinator
from finpilot.api import serializers as ser
from finpilot.api.deps import current_user_id, get_db, require_admin
from finpilot.llm.client import run_context, step_call_log
from finpilot.models import AgentRun, AgentStep, ChatMessage, Digest, LLMCall
from finpilot.schemas import ChatRequest, RunRequest, SchedulerUpdate
from finpilot.services import repository as repo
from finpilot.services.scheduler import scheduler
from finpilot.services.toolkit import FinanceToolkit

router = APIRouter()


@router.get("/runs")
def list_runs(
    limit: int = 30, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    rows = db.scalars(
        select(AgentRun).where(AgentRun.user_id == uid).order_by(AgentRun.id.desc()).limit(min(limit, 200))
    )
    return {"runs": [ser.run(r) for r in rows]}


@router.get("/runs/{run_id}")
def get_run(
    run_id: int, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    r = db.get(AgentRun, run_id)
    if r is None or r.user_id != uid:
        raise HTTPException(404, "run not found")
    calls = db.scalars(select(LLMCall).where(LLMCall.run_id == run_id).order_by(LLMCall.id))
    return {
        "run": ser.run(r, with_steps=True),
        "llm_calls": [
            {
                "id": c.id,
                "agent": c.agent,
                "provider": c.provider,
                "model": c.model,
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "latency_ms": c.latency_ms,
                "cached": c.cached,
                "success": c.success,
                "error": c.error,
                "kind": c.kind or "llm",
                "purpose": c.purpose,
                "cost_usd": c.cost_usd,
            }
            for c in calls
        ],
    }


@router.post("/runs")
async def start_run(body: RunRequest, uid: int = Depends(current_user_id)) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            Coordinator().run_pipeline,
            uid,
            trigger="manual",
            recategorize=body.recategorize,
            deliver_digest=body.deliver_digest,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/chat")
def chat(
    body: ChatRequest, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    history = [
        {"role": m.role, "content": m.content}
        for m in db.scalars(
            select(ChatMessage)
            .where(ChatMessage.user_id == uid, ChatMessage.session_id == body.session_id)
            .order_by(ChatMessage.id.desc())
            .limit(8)
        )
    ][::-1]
    kit = FinanceToolkit(db, uid)
    run = AgentRun(user_id=uid, trigger="chat", graph="chat", status="running")
    db.add(run)
    db.commit()
    t0 = time.perf_counter()
    started = dt.datetime.now(dt.UTC)
    with run_context(run.id), step_call_log() as calls:
        try:
            result = ChatAgent().answer(kit, body.message, history)
            run.status = "succeeded"
        except Exception as exc:  # noqa: BLE001
            run.status, run.error = "failed", str(exc)
            result = {"answer": "Sorry — something went wrong answering that.", "trace": [], "model": "error"}
    route = result.get("route")
    run.finished_at = dt.datetime.now(dt.UTC)
    run.summary = {
        "question": body.message[:200],
        "tools": [t["tool"] for t in result["trace"]],
        "model": result["model"],
        "route": route and {k: route[k] for k in ("intent", "confidence", "action")},
    }
    llm_calls = [c for c in calls if c.get("kind") != "jev"]
    how = f"via {result['model']}"
    if route:
        how = (
            f"routed by Jev as '{route['intent']}' ({route['confidence']:.2f}) → "
            f"{route['action'].replace('_', ' ')}"
            + (f" ({route['llm_model']})" if route.get("llm_model") else "")
        )
    db.add(
        AgentStep(
            run_id=run.id,
            agent="chat",
            status=run.status,
            started_at=started,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            summary=f"Answered with {len(result['trace'])} tool call(s), {how}",
            details={
                "trace": result["trace"],
                "route": route,
                "guard": result.get("guard"),
                "llm_calls": llm_calls or None,
                "jev_calls": [c for c in calls if c.get("kind") == "jev"] or None,
            },
            llm_calls=len(llm_calls),
            error=run.error,
        )
    )
    db.add(ChatMessage(user_id=uid, session_id=body.session_id, role="user", content=body.message))
    msg = ChatMessage(
        user_id=uid,
        session_id=body.session_id,
        role="assistant",
        content=result["answer"],
        trace=result["trace"],
        route=route,
    )
    db.add(msg)
    db.commit()
    return {
        "id": msg.id,
        "answer": result["answer"],
        "trace": result["trace"],
        "model": result["model"],
        "route": route,
        "run_id": run.id,
        "session_id": body.session_id,
    }


@router.get("/chat/{session_id}")
def chat_history(
    session_id: str, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
        .limit(200)
    )
    return {
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "trace": m.trace or [],
                "route": m.route,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in rows
        ]
    }


@router.delete("/chat/{session_id}", status_code=204)
def clear_chat(session_id: str, uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> None:
    for m in db.scalars(
        select(ChatMessage).where(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
    ):
        db.delete(m)
    db.commit()


@router.get("/chat-session")
def new_chat_session() -> dict[str, str]:
    return {"session_id": uuid.uuid4().hex[:16]}


@router.get("/llm/usage")
def llm_usage(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.execute(
        select(
            LLMCall.agent,
            LLMCall.provider,
            LLMCall.model,
            func.count(),
            func.sum(LLMCall.prompt_tokens),
            func.sum(LLMCall.completion_tokens),
            func.avg(LLMCall.latency_ms),
            func.sum(case((LLMCall.cached.is_(True), 1), else_=0)),
            func.sum(case((LLMCall.success.is_(True), 1), else_=0)),
        )
        .where(LLMCall.user_id == uid)
        .group_by(LLMCall.agent, LLMCall.provider, LLMCall.model)
    ).all()
    items = [
        {
            "agent": a,
            "provider": p,
            "model": m,
            "calls": n,
            "prompt_tokens": int(pt or 0),
            "completion_tokens": int(ct or 0),
            "avg_latency_ms": round(float(lat or 0)),
            "cache_hits": int(ch or 0),
            "successes": int(ok or 0),
        }
        for a, p, m, n, pt, ct, lat, ch, ok in rows
    ]
    return {
        "items": items,
        "total_calls": sum(i["calls"] for i in items),
        "total_tokens": sum(i["prompt_tokens"] + i["completion_tokens"] for i in items),
    }


@router.get("/digests")
def list_digests(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.scalars(select(Digest).where(Digest.user_id == uid).order_by(Digest.id.desc()).limit(50))
    return {
        "digests": [ser.digest(d) for d in rows],
        "scheduler": {
            "running": scheduler.running,
            "jobs": scheduler.jobs(),
            "digest_cron": scheduler.digest_cron,
        },
    }


@router.post("/digests/send-now")
async def send_digest_now(uid: int = Depends(current_user_id)) -> dict[str, Any]:
    try:
        res = await run_in_threadpool(Coordinator().run_pipeline, uid, trigger="manual", deliver_digest=True)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return res


@router.get("/scheduler")
def get_scheduler() -> dict[str, Any]:
    return {
        "running": scheduler.running,
        "jobs": scheduler.jobs(),
        "digest_cron": scheduler.digest_cron,
        "pipeline_cron": scheduler.pipeline_cron,
        "timezone": scheduler.settings.timezone,
    }


@router.put("/scheduler", dependencies=[Depends(require_admin)])
def update_scheduler(body: SchedulerUpdate, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        scheduler.reschedule_digest(body.digest_cron)
    except ValueError as exc:
        raise HTTPException(422, f"invalid cron expression: {exc}") from exc
    repo.put_setting(db, "digest_cron", body.digest_cron)
    db.commit()
    return get_scheduler()
