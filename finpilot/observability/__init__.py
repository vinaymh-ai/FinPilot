"""Observability: request/user context, structured logs, raw model-call traces, optional LangSmith."""

from finpilot.observability.context import (
    current_request_id,
    current_user_id,
    new_request_id,
    request_id_var,
    user_context,
    user_id_var,
)
from finpilot.observability.tracing import (
    clip,
    configure_langsmith,
    langsmith_enabled,
    langsmith_status,
    traceable,
    wrap_openai_client,
)

__all__ = [
    "clip",
    "configure_langsmith",
    "current_request_id",
    "current_user_id",
    "langsmith_enabled",
    "langsmith_status",
    "new_request_id",
    "request_id_var",
    "traceable",
    "user_context",
    "user_id_var",
    "wrap_openai_client",
]
