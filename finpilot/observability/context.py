"""Request-scoped context: which request and which user a log line or model call belongs to.

Context variables flow into worker threads (FastAPI's threadpool, the Jev fan-out, LangGraph
nodes), so every log record and every recorded model call can be tied back to the HTTP request
and the user that caused it.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("finpilot_request_id", default=None)
user_id_var: ContextVar[int | None] = ContextVar("finpilot_user_id", default=None)

_VALID_RID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def new_request_id(incoming: str | None = None) -> str:
    """Honour a well-formed X-Request-ID from a proxy; otherwise mint one."""
    if incoming and _VALID_RID.match(incoming):
        return incoming
    return uuid.uuid4().hex[:16]


def current_request_id() -> str | None:
    return request_id_var.get()


def current_user_id() -> int | None:
    return user_id_var.get()


@contextmanager
def user_context(user_id: int | None) -> Iterator[None]:
    token = user_id_var.set(user_id)
    try:
        yield
    finally:
        user_id_var.reset(token)
