"""Tracing: built-in payload capture (always local) and optional LangSmith export.

* **Built-in** — every LLM and Jev call is stored in `llm_calls` with its raw request and response
  (clipped to `TRACE_MAX_CHARS` per field, purged after `TRACE_RETENTION_DAYS`). Nothing leaves
  your database. Browse it on the Observability page or `GET /api/observability/calls/{id}`.
* **LangSmith** (opt-in: `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`) — the LangGraph pipeline
  is traced natively (each agent node a span), OpenAI-SDK calls are wrapped with
  `langsmith.wrappers.wrap_openai`, and Jev decisions / chat turns are `@traceable` spans nested
  under them. Off by default because traces contain financial data and leave your machine.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar

from finpilot.config import Settings, get_settings

log = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])
_enabled = False


def configure_langsmith(settings: Settings | None = None) -> dict[str, Any]:
    """Export LangSmith settings to the environment LangChain/LangGraph read. Returns a status dict."""
    global _enabled
    s = settings or get_settings()
    _enabled = bool(s.langsmith_tracing and s.langsmith_api_key)
    if s.langsmith_tracing and not s.langsmith_api_key:
        log.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is not set — LangSmith tracing stays off")
    if _enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key or ""
        os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
        if s.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = s.langsmith_endpoint
        log.info("LangSmith tracing enabled (project %s)", s.langsmith_project)
    else:
        os.environ.pop("LANGSMITH_TRACING", None)
    return langsmith_status(s)


def langsmith_enabled() -> bool:
    return _enabled


def langsmith_status(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    return {
        "enabled": _enabled,
        "requested": s.langsmith_tracing,
        "key_set": bool(s.langsmith_api_key),
        "project": s.langsmith_project,
        "url": "https://smith.langchain.com/" if _enabled else None,
    }


def wrap_openai_client(client: Any) -> Any:
    if not _enabled:
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception:  # noqa: BLE001 - tracing must never break a model call
        log.warning("could not wrap the OpenAI client for LangSmith", exc_info=True)
        return client


def traceable(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """`langsmith.traceable` when available (a cheap no-op while tracing is off)."""

    def deco(fn: F) -> F:
        try:
            from langsmith import traceable as ls_traceable

            traced = ls_traceable(name=name, run_type=run_type)(fn)
        except Exception:  # noqa: BLE001
            return fn

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return traced(*args, **kwargs) if _enabled else fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return deco


def clip(obj: Any, max_chars: int, _depth: int = 0) -> Any:
    """JSON-safe copy with long strings and lists truncated (for stored traces)."""
    if _depth > 12:
        return "…"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= max_chars else obj[:max_chars] + f"… [+{len(obj) - max_chars} chars]"
    if isinstance(obj, dict):
        return {str(k): clip(v, max_chars, _depth + 1) for k, v in list(obj.items())[:200]}
    if isinstance(obj, (list, tuple)):
        items = [clip(v, max_chars, _depth + 1) for v in list(obj)[:200]]
        if len(obj) > 200:
            items.append(f"… [+{len(obj) - 200} items]")
        return items
    return clip(str(obj), max_chars, _depth + 1)
