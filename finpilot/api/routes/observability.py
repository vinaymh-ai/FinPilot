"""Observability API: model-call traces with raw payloads, a searchable log viewer, client errors.

* `/observability/calls` and `/calls/{id}` — every LLM/Jev call with its raw request and
  response (yours; an admin can see everyone's with `all_users=true`).
* `/observability/summary` — volume, failures, latency percentiles, tokens and cost over a window.
* `/observability/logs` — admin only: the tail of the JSON-lines log file, filterable by level,
  request id, user, event or free text. Logs span every account, hence the restriction.
* `/client-errors` — the browser reports crashes here so they land in the same log.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from finpilot.api.deps import Principal, current_principal, get_db, require_admin
from finpilot.config import get_settings
from finpilot.models import LLMCall
from finpilot.observability.tracing import clip, langsmith_status
from finpilot.schemas import ClientErrorReport
from finpilot.services.auth import limiter

router = APIRouter(prefix="/observability")
public = APIRouter()
client_log = logging.getLogger("finpilot.client")

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_TAIL_BYTES = 8 * 1024 * 1024


def _iso(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value else None


def call_row(c: LLMCall, *, payloads: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": c.id,
        "created_at": _iso(c.created_at),
        "kind": c.kind or "llm",
        "agent": c.agent,
        "purpose": c.purpose,
        "provider": c.provider,
        "model": c.model,
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "latency_ms": c.latency_ms,
        "cached": c.cached,
        "success": c.success,
        "error": c.error,
        "cost_usd": c.cost_usd,
        "run_id": c.run_id,
        "user_id": c.user_id,
        "request_id": c.request_id,
        "has_payload": c.request_payload is not None or c.response_payload is not None,
    }
    if payloads:
        row["request"] = c.request_payload
        row["response"] = c.response_payload
    return row


def _scope(q, principal: Principal, all_users: bool):  # type: ignore[no-untyped-def]
    if all_users and principal.is_admin:
        return q
    return q.where(LLMCall.user_id == principal.user_id)


@router.get("/status")
def status(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    s = get_settings()
    log_file = Path(s.log_file) if s.log_file else None
    out: dict[str, Any] = {
        "langsmith": langsmith_status(s),
        "trace_payloads": s.trace_payloads,
        "trace_max_chars": s.trace_max_chars,
        "trace_retention_days": s.trace_retention_days,
        "log_level": s.log_level,
        "log_file_enabled": bool(log_file),
        "request_id_header": "X-Request-ID",
        "is_admin": principal.is_admin,
    }
    if principal.is_admin and log_file:
        out["log_file"] = str(log_file)
        out["log_file_bytes"] = log_file.stat().st_size if log_file.exists() else 0
    return out


@router.get("/summary")
def summary(
    hours: int = 24,
    all_users: bool = False,
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    hours = min(max(hours, 1), 24 * 90)
    since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
    base = _scope(select(LLMCall).where(LLMCall.created_at >= since), principal, all_users).subquery()
    kinds: dict[str, Any] = {}
    kind_expression = func.coalesce(base.c.kind, "llm")
    for kind, n, ok, cached, tin, tout, cost, avg in db.execute(
        select(
            kind_expression,
            func.count(),
            func.sum(case((base.c.success.is_(True), 1), else_=0)),
            func.sum(case((base.c.cached.is_(True), 1), else_=0)),
            func.sum(base.c.prompt_tokens),
            func.sum(base.c.completion_tokens),
            func.sum(base.c.cost_usd),
            func.avg(base.c.latency_ms),
        ).group_by(kind_expression)
    ):
        kinds[kind] = {
            "calls": n,
            "failures": n - int(ok or 0),
            "cache_hits": int(cached or 0),
            "input_tokens": int(tin or 0),
            "output_tokens": int(tout or 0),
            "cost_usd": round(float(cost or 0), 6),
            "avg_latency_ms": round(float(avg or 0)),
        }
    latencies = sorted(
        v
        for (v,) in db.execute(
            select(base.c.latency_ms).where(base.c.cached.is_not(True), base.c.success.is_(True)).limit(20000)
        )
        if v is not None
    )

    def pct(p: float) -> int | None:
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))] if latencies else None

    by_agent = [
        {"agent": a, "kind": k or "llm", "calls": n, "failures": n - int(ok or 0)}
        for a, k, n, ok in db.execute(
            select(
                base.c.agent,
                base.c.kind,
                func.count(),
                func.sum(case((base.c.success.is_(True), 1), else_=0)),
            )
            .group_by(base.c.agent, base.c.kind)
            .order_by(func.count().desc())
        )
    ]
    errors = [
        {"id": i, "created_at": _iso(t), "agent": a, "kind": k or "llm", "error": (e or "")[:300]}
        for i, t, a, k, e in db.execute(
            select(base.c.id, base.c.created_at, base.c.agent, base.c.kind, base.c.error)
            .where(base.c.success.is_not(True))
            .order_by(base.c.id.desc())
            .limit(10)
        )
    ]
    return {
        "hours": hours,
        "by_kind": kinds,
        "by_agent": by_agent,
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
        "recent_errors": errors,
    }


@router.get("/calls")
def list_calls(
    kind: str | None = None,
    agent: str | None = None,
    success: bool | None = None,
    run_id: int | None = None,
    request_id: str | None = None,
    q: str | None = None,
    before_id: int | None = None,
    limit: int = 50,
    all_users: bool = False,
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    stmt = _scope(select(LLMCall), principal, all_users)
    if kind:
        stmt = stmt.where(func.coalesce(LLMCall.kind, "llm") == kind)
    if agent:
        stmt = stmt.where(LLMCall.agent == agent)
    if success is not None:
        stmt = stmt.where(LLMCall.success.is_(success))
    if run_id is not None:
        stmt = stmt.where(LLMCall.run_id == run_id)
    if request_id:
        stmt = stmt.where(LLMCall.request_id == request_id)
    if q:
        like = f"%{q[:100]}%"
        stmt = stmt.where(
            LLMCall.model.ilike(like)
            | LLMCall.purpose.ilike(like)
            | LLMCall.agent.ilike(like)
            | LLMCall.error.ilike(like)
            | (LLMCall.request_id == q.strip())
        )
    if before_id:
        stmt = stmt.where(LLMCall.id < before_id)
    limit = min(max(limit, 1), 200)
    rows = list(db.scalars(stmt.order_by(LLMCall.id.desc()).limit(limit + 1)))
    return {
        "calls": [call_row(c) for c in rows[:limit]],
        "next_before_id": rows[limit - 1].id if len(rows) > limit else None,
    }


@router.get("/calls/{call_id}")
def get_call(
    call_id: int, principal: Principal = Depends(current_principal), db: Session = Depends(get_db)
) -> dict[str, Any]:
    c = db.get(LLMCall, call_id)
    if c is None or (c.user_id != principal.user_id and not principal.is_admin):
        raise HTTPException(404, "call not found")
    return {"call": call_row(c, payloads=True)}


@router.get("/logs", dependencies=[Depends(require_admin)])
def read_logs(
    level: str = "DEBUG",
    q: str | None = None,
    request_id: str | None = None,
    user_id: int | None = None,
    run_id: int | None = None,
    event: str | None = None,
    logger: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    s = get_settings()
    if not s.log_file:
        return {"enabled": False, "lines": [], "file": None}
    path = Path(s.log_file)
    floor = LEVELS.get(level.upper(), 10)
    needle = (q or "").lower()[:200]
    limit = min(max(limit, 1), 1000)
    out: list[dict[str, Any]] = []
    for file in (path, path.with_name(path.name + ".1")):
        if len(out) >= limit:
            break
        for raw in reversed(_tail(file)):
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if LEVELS.get(str(rec.get("level")), 0) < floor:
                continue
            if request_id and rec.get("request_id") != request_id:
                continue
            if user_id is not None and rec.get("user_id") != user_id:
                continue
            if run_id is not None and rec.get("run_id") != run_id:
                continue
            if event and rec.get("event") != event:
                continue
            if logger and not str(rec.get("logger", "")).startswith(logger):
                continue
            if needle and needle not in raw.lower():
                continue
            out.append(rec)
            if len(out) >= limit:
                break
    return {
        "enabled": True,
        "file": str(path),
        "lines": out,
        "truncated": len(out) >= limit,
    }


def _tail(file: Path) -> list[str]:
    if not file.exists():
        return []
    with file.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        start = max(0, size - _TAIL_BYTES)
        fh.seek(start)
        data = fh.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[1:] if start else lines  # first line may be cut mid-record


@public.post("/client-errors", status_code=202)
def client_error(body: ClientErrorReport, request: Request) -> dict[str, bool]:
    ip = request.client.host if request.client else "?"
    if not limiter.hit(f"client-errors:{ip}", 30, 300):
        return {"accepted": False}
    client_log.error(
        "browser error: %s",
        body.message[:300] or "(no message)",
        extra={
            "event": "client_error",
            "event_id": body.event_id,
            "url": body.url,
            "user_agent": body.user_agent,
            "stack": body.stack,
            "context": clip(body.context, 500),
            "client_ip": ip,
        },
    )
    return {"accepted": True}
