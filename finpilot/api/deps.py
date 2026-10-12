"""FastAPI dependencies."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from finpilot.config import get_settings
from finpilot.db import get_db  # noqa: F401 - re-exported for routers
from finpilot.services.repository import DEFAULT_USER_ID


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Optional bearer-token protection: enabled only when API_TOKEN is set."""
    token = get_settings().api_token
    if not token:
        return
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API token")


def current_user_id() -> int:
    # Single-tenant by default; swap for real auth (JWT/OIDC) in a multi-user deployment.
    return DEFAULT_USER_ID
