"""ASGI middleware: request IDs, access logging with timing and user, security headers."""

from __future__ import annotations

import logging
import time

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from finpilot.observability.context import new_request_id, request_id_var

log = logging.getLogger("finpilot.http")

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "strict-origin-when-cross-origin",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
}
QUIET_PATHS = ("/api/health",)


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        rid = new_request_id(headers.get("x-request-id"))
        scope.setdefault("state", {})["request_id"] = rid
        token = request_id_var.set(rid)
        start = time.perf_counter()
        finished: float | None = None  # response fully sent (background tasks may run after it)
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status, finished
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finished = time.perf_counter()
            if message["type"] == "http.response.start":
                status = message["status"]
                h = MutableHeaders(scope=message)
                h["x-request-id"] = rid
                for k, v in SECURITY_HEADERS.items():
                    h.setdefault(k, v)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            log.exception("unhandled error", extra={"event": "http_error", "path": scope.get("path")})
            raise
        finally:
            path = scope.get("path", "")
            if path.startswith("/api") or status >= 400:
                duration = round(((finished or time.perf_counter()) - start) * 1000, 1)
                level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
                if path in QUIET_PATHS and status < 400:
                    level = logging.DEBUG
                state = scope.get("state") or {}
                client = scope.get("client")
                log.log(
                    level,
                    "%s %s -> %s (%sms)",
                    scope.get("method"),
                    path,
                    status,
                    duration,
                    extra={
                        "event": "http_request",
                        "method": scope.get("method"),
                        "path": path,
                        "status": status,
                        "duration_ms": duration,
                        "user_id": state.get("user_id"),
                        "client_ip": client[0] if client else None,
                    },
                )
            request_id_var.reset(token)
