"""Agent runs (decision trail), chat, LLM usage, digests and the scheduler."""

from __future__ import annotations

import datetime as dt
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from finpilot.agents.chat import ChatAgent
from finpilot.agents.coordinator import Coordinator
from finpilot.agents.memory import memory_rows, title_from_question
from finpilot.api import serializers as ser
from finpilot.api.deps import current_user_id, get_db, require_admin
from finpilot.llm.client import run_context, step_call_log
from finpilot.models import AgentRun, AgentStep, ChatMessage, ChatSession, Digest, LLMCall, UserMemory
from finpilot.schemas import ChatRequest, ChatSessionUpdate, RunRequest, SchedulerUpdate
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


def _session(db: Session, uid: int, session_id: str) -> ChatSession | None:
    return db.scalar(
        select(ChatSession).where(ChatSession.user_id == uid, ChatSession.session_id == session_id)
    )


def _session_dict(cs: ChatSession) -> dict[str, Any]:
    return {
        "id": cs.session_id,
        "title": cs.title,
        "title_source": cs.title_source,
        "pinned": bool(cs.pinned),
        "message_count": cs.message_count or 0,
        "created_at": cs.created_at.isoformat() if cs.created_at else None,
        "updated_at": cs.updated_at.isoformat() if cs.updated_at else None,
    }


def _message_dict(m: ChatMessage) -> dict[str, Any]:
    meta = m.meta or {}
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "trace": m.trace or [],
        "route": m.route,
        "artifacts": m.artifacts or [],
        "model": meta.get("model"),
        "pipeline": meta.get("pipeline") or [],
        "guard": meta.get("guard"),
        "memory": meta.get("memory"),
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


@router.post("/chat")
def chat(
    body: ChatRequest, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    session_id = body.session_id or uuid.uuid4().hex[:16]
    cs = _session(db, uid, session_id)
    if cs is None:
        cs = ChatSession(
            session_id=session_id,
            user_id=uid,
            title=title_from_question(body.message),
            title_source="auto",
            message_count=0,
        )
        db.add(cs)
        db.flush()
    # Working memory: the last turns of THIS conversation, with what each answer was about.
    history = [
        {"role": m.role, "content": m.content, "meta": m.meta or {}}
        for m in db.scalars(
            select(ChatMessage)
            .where(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.desc())
            .limit(24)
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
            db.rollback()
            run.status, run.error = "failed", str(exc)
            result = {"answer": "Sorry — something went wrong answering that.", "trace": [], "model": "error"}
    route = result.get("route")
    run.finished_at = dt.datetime.now(dt.UTC)
    artifacts = result.get("artifacts") or []
    pipeline = result.get("pipeline") or []
    memory = result.get("memory")
    run.summary = {
        "question": body.message[:200],
        "tools": [t["tool"] for t in result["trace"]],
        "model": result["model"],
        "pipeline": pipeline,
        "route": route and {k: route[k] for k in ("intent", "confidence", "action")},
        "memory": memory and {k: memory.get(k) for k in ("kind", "payees", "notes")},
    }
    llm_calls = [c for c in calls if c.get("kind") != "jev"]
    how = f"via {' → '.join(pipeline) or 'rules'} ({result['model']})"
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
                "planner": result.get("planner"),
                "analyst": result.get("analyst"),
                "memory": memory,
                "pipeline": pipeline,
                "llm_calls": llm_calls or None,
                "jev_calls": [c for c in calls if c.get("kind") == "jev"] or None,
            },
            llm_calls=len(llm_calls),
            error=run.error,
        )
    )
    db.add(ChatMessage(user_id=uid, session_id=session_id, role="user", content=body.message))
    meta = {
        "model": result["model"],
        "pipeline": pipeline,
        "guard": result.get("guard"),
        "analyst": result.get("analyst"),
        "planner": result.get("planner"),
        "memory": memory,
        "context": result.get("context"),
    }
    msg = ChatMessage(
        user_id=uid,
        session_id=session_id,
        role="assistant",
        content=result["answer"],
        trace=result["trace"],
        route=route,
        artifacts=artifacts or None,
        meta=meta,
    )
    db.add(msg)
    cs = _session(db, uid, session_id) or cs
    cs.message_count = (cs.message_count or 0) + 2
    cs.updated_at = dt.datetime.now(dt.UTC)
    db.commit()
    return {
        "id": msg.id,
        "answer": result["answer"],
        "trace": result["trace"],
        "model": result["model"],
        "route": route,
        "planner": result.get("planner"),
        "analyst": result.get("analyst"),
        "guard": result.get("guard"),
        "memory": memory,
        "pipeline": pipeline,
        "artifacts": artifacts,
        "run_id": run.id,
        "session_id": session_id,
        "session": _session_dict(cs),
    }


@router.get("/chat/sessions")
def list_chat_sessions(
    q: str | None = None,
    limit: int = 100,
    uid: int = Depends(current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Conversations, most recent first (pinned on top). `q` searches titles and messages."""
    stmt = select(ChatSession).where(ChatSession.user_id == uid)
    if q and q.strip():
        like = f"%{q.strip()[:80]}%"
        matching = select(ChatMessage.session_id).where(
            ChatMessage.user_id == uid, ChatMessage.content.ilike(like)
        )
        stmt = stmt.where(or_(ChatSession.title.ilike(like), ChatSession.session_id.in_(matching)))
    rows = db.scalars(
        stmt.order_by(ChatSession.pinned.desc(), ChatSession.updated_at.desc(), ChatSession.id.desc()).limit(
            min(max(limit, 1), 500)
        )
    )
    return {"sessions": [_session_dict(cs) for cs in rows]}


@router.patch("/chat/sessions/{session_id}")
def update_chat_session(
    session_id: str,
    body: ChatSessionUpdate,
    uid: int = Depends(current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    cs = _session(db, uid, session_id)
    if cs is None:
        raise HTTPException(404, "conversation not found")
    if body.title is not None:
        cs.title, cs.title_source = body.title.strip(), "user"
    if body.pinned is not None:
        cs.pinned = body.pinned
    db.commit()
    return {"session": _session_dict(cs)}


@router.delete("/chat/sessions/{session_id}", status_code=204)
def delete_chat_session(
    session_id: str, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> None:
    db.execute(delete(ChatMessage).where(ChatMessage.user_id == uid, ChatMessage.session_id == session_id))
    db.execute(delete(ChatSession).where(ChatSession.user_id == uid, ChatSession.session_id == session_id))
    db.commit()


@router.get("/chat/memory")
def list_memories(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    """What the chat remembers about this user across conversations."""
    return {"memories": memory_rows(db, uid)}


@router.delete("/chat/memory/{memory_id}", status_code=204)
def delete_memory(memory_id: int, uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> None:
    db.execute(delete(UserMemory).where(UserMemory.user_id == uid, UserMemory.id == memory_id))
    db.commit()


@router.delete("/chat/memory", status_code=204)
def clear_memories(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> None:
    db.execute(delete(UserMemory).where(UserMemory.user_id == uid))
    db.commit()


@router.get("/chat/{session_id}")
def chat_history(
    session_id: str, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
        .limit(500)
    )
    cs = _session(db, uid, session_id)
    return {
        "session": _session_dict(cs) if cs else None,
        "messages": [_message_dict(m) for m in rows],
    }


@router.delete("/chat/{session_id}", status_code=204)
def clear_chat(session_id: str, uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> None:
    """Clear a conversation's messages (the thread itself stays; delete it via /chat/sessions)."""
    db.execute(delete(ChatMessage).where(ChatMessage.user_id == uid, ChatMessage.session_id == session_id))
    cs = _session(db, uid, session_id)
    if cs is not None:
        cs.message_count = 0
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
