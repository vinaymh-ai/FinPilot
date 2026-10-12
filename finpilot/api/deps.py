"""FastAPI dependencies: who is calling, and are they allowed to.

`AUTH_ENABLED=true` (default): requests authenticate with the session cookie set at sign-in, or
`Authorization: Bearer <session token>` for API clients. `AUTH_ENABLED=false`: single-user mode —
everything belongs to user 1, optionally gated by a static `API_TOKEN`.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from finpilot.config import get_settings
from finpilot.db import get_db, session_scope  # noqa: F401 - get_db re-exported for routers
from finpilot.observability.context import user_id_var
from finpilot.services.repository import DEFAULT_USER_ID

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@dataclass(frozen=True)
class Principal:
    user_id: int
    is_admin: bool
    is_demo: bool = False
    via: str = "single_user"  # single_user | cookie | bearer
    token: str | None = None


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization[:7].lower() == "bearer ":
        return authorization[7:].strip() or None
    return None


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Static bearer-token gate for single-user mode: enabled only when API_TOKEN is set."""
    s = get_settings()
    if s.auth_enabled or not s.api_token:
        return
    supplied = _bearer(authorization) or ""
    if not hmac.compare_digest(supplied, s.api_token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API token")


def _lookup(token: str) -> Principal | None:
    from finpilot.services.auth import resolve_session

    with session_scope() as db:
        info = resolve_session(db, token)
        if info is None:
            return None
        u = info.user
        return Principal(user_id=u.id, is_admin=bool(u.is_admin), is_demo=bool(u.is_demo), token=token)


async def current_principal(request: Request, authorization: str | None = Header(default=None)) -> Principal:
    """Resolve the caller once per request. Async so the user context it sets is visible to the
    (threadpool) endpoint, to the access log and to every model call the request makes."""
    cached = getattr(request.state, "principal", None)
    if cached is not None:
        return cached
    s = get_settings()
    if not s.auth_enabled:
        principal = Principal(user_id=DEFAULT_USER_ID, is_admin=True)
    else:
        from finpilot.services.auth import SESSION_COOKIE

        bearer = _bearer(authorization)
        token = bearer or request.cookies.get(SESSION_COOKIE)
        found = await run_in_threadpool(_lookup, token) if token else None
        if found is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                {"message": "Sign in to continue.", "code": "not_authenticated"},
            )
        principal = Principal(
            user_id=found.user_id,
            is_admin=found.is_admin,
            is_demo=found.is_demo,
            via="bearer" if bearer else "cookie",
            token=token,
        )
        # CSRF: browsers attach cookies to cross-site requests but cannot add custom headers to
        # them without a CORS preflight, which our CORS policy refuses for unknown origins.
        if (
            principal.via == "cookie"
            and request.method in UNSAFE_METHODS
            and not request.headers.get("x-requested-with")
        ):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, {"message": "Missing X-Requested-With header.", "code": "csrf"}
            )
    request.state.principal = principal
    request.state.user_id = principal.user_id
    user_id_var.set(principal.user_id)
    return principal


async def current_user_id(principal: Principal = Depends(current_principal)) -> int:
    return principal.user_id


async def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    """Server-wide settings (model providers, Jev, schedules) and raw logs span every account."""
    if not principal.is_admin:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, {"message": "Only an administrator can do this.", "code": "admin_only"}
        )
    return principal
