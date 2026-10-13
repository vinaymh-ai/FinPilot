"""Persist and log one model call (LLM or Jev) with its raw request and response.

Both the LLM router and the Jev client funnel through `record_model_call`, so every call — cached
or live, successful or failed — lands in `llm_calls` tagged with the agent run, the user and the
HTTP request that caused it, and produces one structured log line.
"""

from __future__ import annotations

import logging
from typing import Any

from finpilot.config import get_settings
from finpilot.observability.context import current_request_id, current_user_id
from finpilot.observability.tracing import clip

log = logging.getLogger("finpilot.model_calls")


def record_model_call(
    entry: dict[str, Any],
    *,
    run_id: int | None,
    error: str | None = None,
    purpose: str | None = None,
    request: Any = None,
    response: Any = None,
) -> None:
    s = get_settings()
    user_id = current_user_id()
    request_id = current_request_id()
    level = (
        logging.WARNING
        if not entry.get("success")
        else logging.DEBUG
        if entry.get("cached")
        else logging.INFO
    )
    log.log(
        level,
        "%s call %s/%s %s:%s -> %s",
        entry.get("kind", "llm"),
        entry.get("agent"),
        purpose or "-",
        entry.get("provider"),
        entry.get("model"),
        "ok" if entry.get("success") else f"failed: {error}",
        extra={
            "event": "model_call",
            "kind": entry.get("kind"),
            "agent": entry.get("agent"),
            "purpose": purpose,
            "provider": entry.get("provider"),
            "model": entry.get("model"),
            "prompt_tokens": entry.get("prompt_tokens"),
            "completion_tokens": entry.get("completion_tokens"),
            "latency_ms": entry.get("latency_ms"),
            "cached": entry.get("cached"),
            "success": entry.get("success"),
            "cost_usd": entry.get("cost_usd"),
            "error": error,
        },
    )
    if log.isEnabledFor(logging.DEBUG) and (request is not None or response is not None):
        log.debug(
            "raw %s payload",
            entry.get("kind", "llm"),
            extra={
                "event": "model_payload",
                "request": clip(request, s.trace_max_chars),
                "response": clip(response, s.trace_max_chars),
            },
        )

    from finpilot.db import session_scope
    from finpilot.models import LLMCall

    payloads: dict[str, Any] = {}
    if s.trace_payloads:
        payloads = {
            "request_payload": clip(request, s.trace_max_chars),
            "response_payload": clip(response, s.trace_max_chars),
        }
    try:
        with session_scope() as db:
            db.add(
                LLMCall(
                    run_id=run_id,
                    error=error,
                    purpose=(purpose or None) and purpose[:64],
                    user_id=user_id,
                    request_id=request_id,
                    **payloads,
                    **entry,
                )
            )
    except Exception:  # noqa: BLE001 - telemetry must never break the call itself
        log.debug("failed to record model call", exc_info=True)
