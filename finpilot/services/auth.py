"""Accounts: e-mail + password sign-in, server-side sessions, password reset, demo accounts.

Design choices (all standard library, no extra dependencies):

* **Passwords** are hashed with scrypt (memory-hard, `hashlib.scrypt`), a per-user random salt and
  parameters stored alongside the hash so they can be raised later without breaking old logins.
* **Sessions** are opaque random tokens. Only their SHA-256 is stored, so a leaked database cannot
  be replayed as a login. Browsers get the token in an HttpOnly, SameSite=Lax cookie; API clients
  can send it as `Authorization: Bearer <token>`. Logout and password changes revoke server-side.
* **Password reset** uses single-use, hashed, one-hour tokens. Without SMTP configured the link is
  written to the server log (development mode) instead of being e-mailed.
* **Demo accounts** give every visitor a private, seeded sandbox that expires after
  `DEMO_TTL_HOURS`, so trying FinPilot never mixes data between people.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import logging
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from finpilot.config import get_settings
from finpilot.models import AuthSession, AuthToken, Decision, LLMCall, User
from finpilot.services import repository as repo

log = logging.getLogger("finpilot.auth")

SESSION_COOKIE = "finpilot_session"
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _DKLEN = 2**14, 8, 1, 32
_EMAIL = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9-]+)+$"
)
DEMO_DOMAIN = "demo.finpilot.local"
RESET_TTL = dt.timedelta(hours=1)
_TOUCH_EVERY = dt.timedelta(minutes=5)


class AuthError(ValueError):
    """A user-facing authentication problem (bad credentials, weak password, …)."""

    def __init__(self, message: str, status: int = 400, code: str = "auth_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    # SQLite hands back naive datetimes even for timezone=True columns
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value


def _sha256(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ============================================================================ passwords


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DKLEN)
    b64 = base64.urlsafe_b64encode
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt).decode()}${b64(key).decode()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        algo, n, r, p, salt_b64, key_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_b64)
        expected = base64.urlsafe_b64decode(key_b64)
        key = hashlib.scrypt(
            password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected), maxmem=2**26
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(key, expected)


# A fixed hash to verify against when the e-mail is unknown, so response time does not reveal
# which addresses have accounts.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def normalize_email(email: str) -> str:
    e = (email or "").strip().lower()
    if len(e) > 254 or not _EMAIL.match(e):
        raise AuthError("Enter a valid e-mail address.", code="invalid_email")
    return e


def check_password_strength(password: str, email: str = "") -> None:
    s = get_settings()
    if len(password or "") < s.password_min_length:
        raise AuthError(
            f"Password must be at least {s.password_min_length} characters.", code="weak_password"
        )
    if len(password) > 256:
        raise AuthError("Password is too long (max 256 characters).", code="weak_password")
    if password.lower() in {email.lower(), email.split("@")[0].lower()} or len(set(password)) < 4:
        raise AuthError("Choose a less guessable password.", code="weak_password")


# ============================================================================ rate limiting


class RateLimiter:
    """In-process sliding-window limiter (per key). Good for a single API process; put a shared
    limiter (Redis, the reverse proxy) in front of multi-replica deployments."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float) -> bool:
        """Record an attempt; False when the key is over its limit."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(self._hits) > 10000:  # bound memory under a flood of distinct keys
                for k in [k for k, v in self._hits.items() if not v][:5000]:
                    self._hits.pop(k, None)
            return True

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


limiter = RateLimiter()


def throttle(key: str, limit: int, window_s: float) -> None:
    if not limiter.hit(key, limit, window_s):
        raise AuthError(
            "Too many attempts. Wait a few minutes and try again.", status=429, code="rate_limited"
        )


# ============================================================================ accounts


def _is_admin_email(email: str) -> bool:
    return email in get_settings().admin_email_list


def _credentialed_users(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(User).where(User.password_hash.is_not(None))) or 0


def _legacy_owner(db: Session) -> User | None:
    """User 1 from single-user mode: has data but no login. The first sign-up takes it over, so
    switching AUTH_ENABLED on does not orphan an existing database."""
    user = db.get(User, repo.DEFAULT_USER_ID)
    if user is None or user.password_hash or user.is_demo:
        return None
    return user


def signup(db: Session, email: str, password: str, name: str | None = None) -> User:
    s = get_settings()
    if not s.allow_signup:
        raise AuthError("Sign-ups are disabled on this server.", status=403, code="signup_disabled")
    email = normalize_email(email)
    if email.endswith("@" + DEMO_DOMAIN):
        raise AuthError("That e-mail domain is reserved.", code="invalid_email")
    check_password_strength(password, email)
    if db.scalar(select(User).where(func.lower(User.email) == email)) is not None:
        raise AuthError(
            "An account with this e-mail already exists. Sign in instead.", status=409, code="exists"
        )
    first = _credentialed_users(db) == 0
    user = _legacy_owner(db) if first else None
    claimed = user is not None
    if user is None:
        user = User(name=(name or email.split("@")[0])[:120], currency=s.currency, current_balance=0.0)
        db.add(user)
    elif name:
        user.name = name[:120]
    user.email = email
    user.password_hash = hash_password(password)
    user.is_admin = _is_admin_email(email) or (first and not s.admin_email_list)
    user.is_demo = False
    user.disabled = False
    db.commit()
    log.info(
        "account created for user %s%s%s",
        user.id,
        " (took over existing single-user data)" if claimed else "",
        " [admin]" if user.is_admin else "",
        extra={"event": "signup", "user_id": user.id, "admin": bool(user.is_admin)},
    )
    return user


def authenticate(db: Session, email: str, password: str) -> User:
    try:
        email = normalize_email(email)
    except AuthError:
        verify_password(password, _DUMMY_HASH)
        raise AuthError("Incorrect e-mail or password.", status=401, code="bad_credentials") from None
    user = db.scalar(select(User).where(func.lower(User.email) == email))
    if user is None or not user.password_hash:
        verify_password(password, _DUMMY_HASH)
        raise AuthError("Incorrect e-mail or password.", status=401, code="bad_credentials")
    if not verify_password(password, user.password_hash):
        log.warning(
            "failed sign-in for user %s", user.id, extra={"event": "login_failed", "user_id": user.id}
        )
        raise AuthError("Incorrect e-mail or password.", status=401, code="bad_credentials")
    if user.disabled:
        raise AuthError("This account is disabled.", status=403, code="disabled")
    if _is_admin_email(email) and not user.is_admin:
        user.is_admin = True
    user.last_login_at = _now()
    db.commit()
    return user


def change_password(
    db: Session, user: User, current: str, new: str, *, keep_session: str | None = None
) -> None:
    if not verify_password(current, user.password_hash):
        raise AuthError("Current password is incorrect.", status=401, code="bad_credentials")
    check_password_strength(new, user.email or "")
    user.password_hash = hash_password(new)
    revoke_all_sessions(db, user.id, keep_token=keep_session)
    db.commit()
    log.info("password changed for user %s", user.id, extra={"event": "password_changed", "user_id": user.id})


# ============================================================================ sessions


@dataclass
class SessionInfo:
    user: User
    token_hash: str


def create_session(db: Session, user: User, *, user_agent: str | None = None, ip: str | None = None) -> str:
    token = secrets.token_urlsafe(32)
    ttl = dt.timedelta(days=get_settings().session_ttl_days)
    if user.is_demo:
        ttl = min(ttl, dt.timedelta(hours=get_settings().demo_ttl_hours))
    db.add(
        AuthSession(
            token_hash=_sha256(token),
            user_id=user.id,
            expires_at=_now() + ttl,
            last_seen_at=_now(),
            user_agent=(user_agent or "")[:255] or None,
            ip=(ip or "")[:64] or None,
        )
    )
    db.commit()
    log.info(
        "signed in user %s", user.id, extra={"event": "login", "user_id": user.id, "demo": bool(user.is_demo)}
    )
    return token


def resolve_session(db: Session, token: str | None) -> SessionInfo | None:
    if not token or len(token) > 200:
        return None
    th = _sha256(token)
    row = db.get(AuthSession, th)
    if row is None:
        return None
    now = _now()
    if (_aware(row.expires_at) or now) <= now:
        db.delete(row)
        db.commit()
        return None
    user = db.get(User, row.user_id)
    if user is None or user.disabled:
        return None
    last = _aware(row.last_seen_at)
    if last is None or now - last > _TOUCH_EVERY:
        row.last_seen_at = now
        db.commit()
    return SessionInfo(user=user, token_hash=th)


def revoke_session(db: Session, token: str | None) -> None:
    if token:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == _sha256(token)))
        db.commit()


def revoke_all_sessions(db: Session, user_id: int, *, keep_token: str | None = None) -> None:
    q = delete(AuthSession).where(AuthSession.user_id == user_id)
    if keep_token:
        q = q.where(AuthSession.token_hash != _sha256(keep_token))
    db.execute(q)


# ============================================================================ password reset


def create_reset_token(db: Session, email: str) -> tuple[User, str] | None:
    """Returns (user, token) or None when there is no such account (callers must not reveal which)."""
    try:
        email = normalize_email(email)
    except AuthError:
        return None
    user = db.scalar(select(User).where(func.lower(User.email) == email))
    if user is None or not user.password_hash or user.disabled or user.is_demo:
        return None
    token = secrets.token_urlsafe(32)
    db.add(
        AuthToken(
            token_hash=_sha256(token),
            user_id=user.id,
            purpose="password_reset",
            expires_at=_now() + RESET_TTL,
        )
    )
    db.commit()
    return user, token


def reset_password(db: Session, token: str, new_password: str) -> User:
    row = db.get(AuthToken, _sha256(token or ""))
    now = _now()
    if (
        row is None
        or row.purpose != "password_reset"
        or row.used_at is not None
        or (_aware(row.expires_at) or now) <= now
    ):
        raise AuthError("This reset link is invalid or has expired. Request a new one.", code="invalid_token")
    user = db.get(User, row.user_id)
    if user is None or user.disabled:
        raise AuthError("This reset link is invalid or has expired. Request a new one.", code="invalid_token")
    check_password_strength(new_password, user.email or "")
    user.password_hash = hash_password(new_password)
    row.used_at = now
    revoke_all_sessions(db, user.id)
    db.commit()
    log.info("password reset for user %s", user.id, extra={"event": "password_reset", "user_id": user.id})
    return user


# ============================================================================ demo accounts


def create_demo_user(db: Session) -> User:
    s = get_settings()
    if not s.demo_accounts:
        raise AuthError("Demo accounts are disabled on this server.", status=403, code="demo_disabled")
    from finpilot.services.seed import seed_demo

    user = User(
        name="Demo User",
        email=f"demo-{secrets.token_hex(6)}@{DEMO_DOMAIN}",
        currency=s.currency,
        current_balance=0.0,
        is_demo=True,
        digest_enabled=False,
    )
    db.add(user)
    db.commit()
    seed_demo(db, user.id)
    log.info("demo account %s created", user.id, extra={"event": "demo_created", "user_id": user.id})
    return user


def delete_user(db: Session, user_id: int) -> None:
    """Remove an account and everything it owns (foreign keys cascade; telemetry rows are plain ints)."""
    repo.wipe_user_data(db, user_id)
    db.execute(delete(LLMCall).where(LLMCall.user_id == user_id))
    db.execute(delete(Decision).where(Decision.user_id == user_id))
    db.execute(delete(User).where(User.id == user_id))
    db.commit()


# ============================================================================ maintenance


def purge_expired(db: Session) -> dict[str, int]:
    now = _now()
    sessions = db.execute(delete(AuthSession).where(AuthSession.expires_at < now)).rowcount or 0
    tokens = (
        db.execute(delete(AuthToken).where(AuthToken.expires_at < now - dt.timedelta(days=1))).rowcount or 0
    )
    db.commit()
    cutoff = now - dt.timedelta(hours=get_settings().demo_ttl_hours)
    demo_ids = list(db.scalars(select(User.id).where(User.is_demo.is_(True), User.created_at < cutoff)))
    for uid in demo_ids:
        delete_user(db, uid)
    return {"sessions": sessions, "tokens": tokens, "demo_accounts": len(demo_ids)}


def public_user(user: User) -> dict[str, object]:
    return {
        "id": user.id,
        "email": None if user.is_demo else user.email,
        "name": user.name,
        "is_admin": bool(user.is_admin),
        "is_demo": bool(user.is_demo),
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }
