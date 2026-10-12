"""Accounts: sign up, sign in, sign out, demo accounts, password reset and change, account deletion."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from finpilot.api.deps import Principal, current_principal
from finpilot.config import get_settings
from finpilot.db import session_scope
from finpilot.models import User
from finpilot.schemas import (
    ChangePasswordRequest,
    DeleteAccountRequest,
    ForgotPasswordRequest,
    LoginRequest,
    ResetPasswordRequest,
    SignupRequest,
)
from finpilot.services import auth
from finpilot.services import repository as repo

log = logging.getLogger("finpilot.auth")

public = APIRouter(prefix="/auth")
router = APIRouter(prefix="/auth")


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "?"


def _require_xhr(request: Request) -> None:
    # Same CSRF rule as authenticated writes: stops a third-party page from silently signing a
    # visitor in (or out) of an account it controls.
    if not request.headers.get("x-requested-with"):
        raise HTTPException(403, {"message": "Missing X-Requested-With header.", "code": "csrf"})


def _secure(request: Request) -> bool:
    s = get_settings()
    if s.cookie_secure is not None:
        return s.cookie_secure
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto.split(",")[0].strip() == "https"


def _signed_in(
    request: Request, user: User, token: str, *, remember: bool = True, issue_token: bool = False
) -> JSONResponse:
    s = get_settings()
    hours = s.demo_ttl_hours if user.is_demo else s.session_ttl_days * 24
    body: dict[str, Any] = {"user": auth.public_user(user)}
    if issue_token:
        body["token"] = token
    resp = JSONResponse(body)
    resp.set_cookie(
        auth.SESSION_COOKIE,
        token,
        max_age=int(hours * 3600) if remember else None,
        httponly=True,
        secure=_secure(request),
        samesite="lax",
        path="/",
    )
    return resp


def _fail(exc: auth.AuthError) -> HTTPException:
    return HTTPException(exc.status, {"message": str(exc), "code": exc.code})


# ---------------------------------------------------------------------------- public


@public.get("/config")
def auth_config() -> dict[str, Any]:
    s = get_settings()
    return {
        "auth_enabled": s.auth_enabled,
        "allow_signup": s.allow_signup,
        "demo_accounts": s.demo_accounts,
        "demo_ttl_hours": s.demo_ttl_hours,
        "password_min_length": s.password_min_length,
        "password_reset_by_email": bool(s.smtp_host and s.smtp_user and s.smtp_password),
    }


@public.post("/signup", dependencies=[Depends(_require_xhr)])
def signup(body: SignupRequest, request: Request) -> JSONResponse:
    _auth_mode_only()
    ip = _client_ip(request)
    try:
        auth.throttle(f"signup:ip:{ip}", 10, 3600)
        with session_scope() as db:
            user = auth.signup(db, body.email, body.password, body.name)
            token = auth.create_session(db, user, user_agent=request.headers.get("user-agent"), ip=ip)
            return _signed_in(request, user, token, issue_token=body.issue_token)
    except auth.AuthError as exc:
        raise _fail(exc) from exc


@public.post("/login", dependencies=[Depends(_require_xhr)])
def login(body: LoginRequest, request: Request) -> JSONResponse:
    _auth_mode_only()
    ip = _client_ip(request)
    try:
        auth.throttle(f"login:ip:{ip}", 30, 300)
        auth.throttle(f"login:email:{body.email.strip().lower()}", 10, 900)
        with session_scope() as db:
            user = auth.authenticate(db, body.email, body.password)
            token = auth.create_session(db, user, user_agent=request.headers.get("user-agent"), ip=ip)
            return _signed_in(request, user, token, remember=body.remember, issue_token=body.issue_token)
    except auth.AuthError as exc:
        raise _fail(exc) from exc


@public.post("/demo", dependencies=[Depends(_require_xhr)])
def start_demo(request: Request, background: BackgroundTasks) -> JSONResponse:
    """A private, pre-filled account for trying FinPilot; deleted after DEMO_TTL_HOURS."""
    _auth_mode_only()
    ip = _client_ip(request)
    try:
        auth.throttle(f"demo:ip:{ip}", 10, 3600)
        # each demo runs the agents (model spend): cap creation server-wide too
        auth.throttle("demo:all", 120, 3600)
        with session_scope() as db:
            user = auth.create_demo_user(db)
            token = auth.create_session(db, user, user_agent=request.headers.get("user-agent"), ip=ip)
            resp = _signed_in(request, user, token)
    except auth.AuthError as exc:
        raise _fail(exc) from exc
    background.add_task(_analyse, user.id)
    resp.background = background
    return resp


def _analyse(user_id: int) -> None:
    from finpilot.agents.coordinator import Coordinator

    try:
        Coordinator().run_pipeline(user_id, trigger="manual")
    except Exception:  # noqa: BLE001
        log.exception("initial analysis of demo account %s failed", user_id)


@public.post("/logout", dependencies=[Depends(_require_xhr)])
def logout(request: Request, response: Response) -> dict[str, bool]:
    token = request.cookies.get(auth.SESSION_COOKIE)
    authz = request.headers.get("authorization") or ""
    if authz[:7].lower() == "bearer ":
        token = authz[7:].strip()
    if token:
        with session_scope() as db:
            auth.revoke_session(db, token)
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return {"ok": True}


@public.post("/password/forgot", dependencies=[Depends(_require_xhr)])
def forgot_password(body: ForgotPasswordRequest, request: Request) -> dict[str, Any]:
    _auth_mode_only()
    ip = _client_ip(request)
    try:
        auth.throttle(f"forgot:ip:{ip}", 10, 900)
        auth.throttle(f"forgot:email:{body.email.strip().lower()}", 3, 3600)
    except auth.AuthError as exc:
        raise _fail(exc) from exc
    with session_scope() as db:
        found = auth.create_reset_token(db, body.email)
        if found:
            user, token = found
            _send_reset(request, user, token)
    # Same answer whether or not the address has an account.
    return {"ok": True, "message": "If an account exists for that e-mail, a reset link is on its way."}


def _send_reset(request: Request, user: User, token: str) -> None:
    from finpilot.services.notifier import send_email

    s = get_settings()
    base = (s.public_base_url or str(request.base_url)).rstrip("/")
    link = f"{base}/reset-password?token={token}"
    text = (
        f"Someone asked to reset the password for your FinPilot account ({user.email}).\n\n"
        f"Open this link within one hour to choose a new password:\n{link}\n\n"
        "If this wasn't you, ignore this e-mail — your password stays the same."
    )
    html = (
        f"<p>Someone asked to reset the password for your FinPilot account ({user.email}).</p>"
        f'<p><a href="{link}">Choose a new password</a> (valid for one hour).</p>'
        "<p>If this wasn't you, ignore this e-mail — your password stays the same.</p>"
    )
    result = send_email(user.email or "", "Reset your FinPilot password", text, html)
    if result.get("status") == "skipped":
        # Development fallback: no SMTP configured, so the operator relays the link.
        log.warning(
            "SMTP is not configured — password reset link for user %s: %s",
            user.id,
            link,
            extra={"event": "password_reset_link", "user_id": user.id},
        )
    else:
        log.info(
            "password reset e-mail %s for user %s", result.get("status"), user.id, extra={"user_id": user.id}
        )


@public.post("/password/reset", dependencies=[Depends(_require_xhr)])
def reset_password(body: ResetPasswordRequest, request: Request) -> JSONResponse:
    _auth_mode_only()
    try:
        auth.throttle(f"reset:ip:{_client_ip(request)}", 20, 900)
        with session_scope() as db:
            user = auth.reset_password(db, body.token, body.password)
            token = auth.create_session(
                db, user, user_agent=request.headers.get("user-agent"), ip=_client_ip(request)
            )
            return _signed_in(request, user, token)
    except auth.AuthError as exc:
        raise _fail(exc) from exc


def _auth_mode_only() -> None:
    if not get_settings().auth_enabled:
        raise HTTPException(
            404,
            {
                "message": "Accounts are disabled (AUTH_ENABLED=false): single-user mode.",
                "code": "auth_disabled",
            },
        )


@public.get("/session")
async def current_session(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Who is signed in, or `user: null` — lets the app check without a 401 on every visit."""
    try:
        principal = await current_principal(request, authorization)
    except HTTPException:
        return {"user": None, "auth_enabled": get_settings().auth_enabled, "demo_expires_hours": None}
    return await run_in_threadpool(me, principal)


# ---------------------------------------------------------------------------- signed in


@router.get("/me")
def me(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    s = get_settings()
    with session_scope() as db:
        user = repo.get_or_create_user(db, principal.user_id)
        data = auth.public_user(user)
    if not s.auth_enabled:
        data["is_admin"] = True
    return {
        "user": data,
        "auth_enabled": s.auth_enabled,
        "demo_expires_hours": s.demo_ttl_hours if principal.is_demo else None,
    }


@router.post("/password/change")
def change_password(
    body: ChangePasswordRequest, request: Request, principal: Principal = Depends(current_principal)
) -> dict[str, bool]:
    _auth_mode_only()
    try:
        auth.throttle(f"change:user:{principal.user_id}", 10, 900)
        with session_scope() as db:
            user = db.get(User, principal.user_id)
            if user is None or user.is_demo:
                raise auth.AuthError("Demo accounts have no password.", code="demo")
            auth.change_password(
                db, user, body.current_password, body.new_password, keep_session=principal.token
            )
    except auth.AuthError as exc:
        raise _fail(exc) from exc
    return {"ok": True}


@router.post("/sessions/revoke-others")
def revoke_other_sessions(principal: Principal = Depends(current_principal)) -> dict[str, bool]:
    _auth_mode_only()
    with session_scope() as db:
        auth.revoke_all_sessions(db, principal.user_id, keep_token=principal.token)
    return {"ok": True}


@router.post("/account/delete")
async def delete_account(
    body: DeleteAccountRequest, response: Response, principal: Principal = Depends(current_principal)
) -> dict[str, bool]:
    """Permanently delete the signed-in account and all of its data."""
    _auth_mode_only()

    def run() -> None:
        with session_scope() as db:
            user = db.get(User, principal.user_id)
            if user is None:
                return
            if not user.is_demo and not auth.verify_password(body.password or "", user.password_hash):
                raise auth.AuthError("Password is incorrect.", status=401, code="bad_credentials")
            auth.delete_user(db, user.id)
        log.info("account %s deleted", principal.user_id, extra={"event": "account_deleted"})

    try:
        await run_in_threadpool(run)
    except auth.AuthError as exc:
        raise _fail(exc) from exc
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return {"ok": True}
