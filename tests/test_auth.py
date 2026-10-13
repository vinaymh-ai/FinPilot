"""Accounts: sign-up/sign-in, sessions, CSRF, isolation between users, password reset, demo accounts."""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from finpilot.main import create_app
from finpilot.services import auth

XHR = {"X-Requested-With": "FinPilot"}
PASSWORD = "correct horse battery"


@pytest.fixture
def app(accounts):
    return create_app()


def browser(app) -> TestClient:
    """A separate cookie jar per 'person'. No lifespan: the test database is already initialised."""
    return TestClient(app, headers=XHR)


def signup(c: TestClient, email: str, password: str = PASSWORD, **extra):
    return c.post("/api/auth/signup", json={"email": email, "password": password, **extra})


# ---------------------------------------------------------------------------- passwords


def test_password_hashing_round_trip():
    h = auth.hash_password("s3cret-pass")
    assert h.startswith("scrypt$") and "s3cret-pass" not in h
    assert auth.verify_password("s3cret-pass", h)
    assert not auth.verify_password("s3cret-pasS", h)
    assert not auth.verify_password("anything", None)
    assert not auth.verify_password("anything", "md5$garbage")
    assert auth.hash_password("same") != auth.hash_password("same")  # salted


@pytest.mark.parametrize(
    "email,password,code",
    [
        ("not-an-email", PASSWORD, "invalid_email"),
        ("a@b.co", "short", "weak_password"),
        ("asha@example.com", "asha@example.com", "weak_password"),
        ("a@b.co", "aaaaaaaaaaaa", "weak_password"),
    ],
)
def test_signup_validation(app, email, password, code):
    r = signup(browser(app), email, password)
    assert r.status_code == 400 and r.json()["detail"]["code"] == code


# ---------------------------------------------------------------------------- sessions


def test_signup_sign_in_sign_out(app):
    c = browser(app)
    r = signup(c, "Asha@Example.com", name="Asha")
    assert r.status_code == 200, r.text
    user = r.json()["user"]
    assert user["email"] == "asha@example.com" and user["name"] == "Asha"
    assert user["is_admin"] is True  # first account administers the server
    assert "token" not in r.json()  # browsers get an HttpOnly cookie, not a readable token
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "finpilot_session=" in cookie

    assert c.get("/api/auth/me").json()["user"]["email"] == "asha@example.com"
    assert c.get("/api/auth/session").json()["user"]["email"] == "asha@example.com"
    assert c.get("/api/dashboard").status_code == 200

    assert c.post("/api/auth/logout").json()["ok"]
    assert c.get("/api/auth/me").status_code == 401

    anon = browser(app)
    assert (
        anon.post("/api/auth/login", json={"email": "asha@example.com", "password": "nope-nope"}).status_code
        == 401
    )
    r = anon.post("/api/auth/login", json={"email": "ASHA@example.com ", "password": PASSWORD})
    assert r.status_code == 200 and anon.get("/api/auth/me").status_code == 200


def test_unauthenticated_requests_are_rejected_but_public_endpoints_work(app):
    c = browser(app)
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/auth/session").json() == {
        "user": None,
        "auth_enabled": True,
        "demo_expires_hours": None,
    }
    assert c.get("/api/auth/config").json()["auth_enabled"] is True
    for path in ("/api/dashboard", "/api/transactions", "/api/observability/calls", "/api/graph"):
        r = c.get(path)
        assert r.status_code == 401, path
        assert r.json()["detail"]["code"] == "not_authenticated"
    assert c.get("/api/dashboard", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_cookie_writes_need_the_csrf_header(app):
    c = browser(app)
    signup(c, "asha@example.com")
    body = {"date": "2026-10-01", "description": "Coffee", "amount": -150}
    bare = TestClient(app, cookies=c.cookies)  # same session, no X-Requested-With
    r = bare.post("/api/transactions", json=body)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "csrf"
    assert bare.get("/api/transactions").status_code == 200  # reads are fine
    assert c.post("/api/transactions", json=body).status_code == 201
    # sign-in itself is protected too (login CSRF)
    assert (
        bare.post("/api/auth/login", json={"email": "asha@example.com", "password": PASSWORD}).status_code
        == 403
    )


def test_bearer_token_for_api_clients(app):
    c = browser(app)
    token = signup(c, "bot@example.com", issue_token=True).json()["token"]
    api = TestClient(app)  # no cookies, no XHR header: bearer auth is not CSRF-able
    h = {"Authorization": f"Bearer {token}"}
    assert api.get("/api/auth/me", headers=h).json()["user"]["email"] == "bot@example.com"
    assert (
        api.post(
            "/api/transactions", headers=h, json={"date": "2026-10-01", "description": "Tea", "amount": -40}
        ).status_code
        == 201
    )


def test_duplicate_email_and_disabled_signup(app, accounts):
    c = browser(app)
    assert signup(c, "asha@example.com").status_code == 200
    r = signup(browser(app), "ASHA@example.com")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "exists"
    accounts.allow_signup = False
    assert signup(browser(app), "ravi@example.com").status_code == 403


def test_login_is_rate_limited(app):
    signup(browser(app), "asha@example.com")
    c = browser(app)
    codes = [
        c.post("/api/auth/login", json={"email": "asha@example.com", "password": f"wrong-{i}"}).status_code
        for i in range(12)
    ]
    assert codes[:10] == [401] * 10 and codes[-1] == 429
    # even the right password is refused while throttled
    assert (
        c.post("/api/auth/login", json={"email": "asha@example.com", "password": PASSWORD}).status_code == 429
    )


# ---------------------------------------------------------------------------- isolation


def test_users_never_see_each_others_data(app):
    a, b = browser(app), browser(app)
    signup(a, "asha@example.com")
    signup(b, "ravi@example.com")
    assert b.get("/api/auth/me").json()["user"]["is_admin"] is False
    txn = a.post(
        "/api/transactions", json={"date": "2026-10-01", "description": "Rent", "amount": -25000}
    ).json()
    tid = txn["transaction"]["id"] if "transaction" in txn else txn["id"]

    assert b.get("/api/transactions").json()["total"] == 0
    assert a.get("/api/transactions").json()["total"] == 1
    assert b.patch(f"/api/transactions/{tid}", json={"category": "dining"}).status_code == 404
    assert b.get("/api/dashboard").status_code == 200


def test_server_wide_settings_are_admin_only(app):
    admin, user = browser(app), browser(app)
    signup(admin, "owner@example.com")
    signup(user, "guest@example.com")
    for c, ok in ((user, False), (admin, True)):
        r1 = c.put("/api/settings/llm", json={"enabled": False})
        r2 = c.put("/api/settings/decisions", json={"accept_confidence": 0.8})
        r3 = c.put("/api/scheduler", json={"digest_cron": "0 9 * * 1"})
        r4 = c.get("/api/observability/logs")
        for r in (r1, r2, r3, r4):
            assert (r.status_code < 400) is ok, (r.request.url, r.status_code, r.text)
    # but everyone can manage their own profile
    assert user.put("/api/settings/profile", json={"name": "Guest"}).status_code == 200


def test_admin_emails_setting(app, accounts):
    accounts.admin_emails = "boss@example.com"
    first = browser(app)
    assert signup(first, "first@example.com").json()["user"]["is_admin"] is False
    assert signup(browser(app), "Boss@Example.com").json()["user"]["is_admin"] is True


def test_first_account_takes_over_single_user_data(app, seeded):
    """Turning accounts on for an existing database must not orphan its data."""
    c = browser(app)
    r = signup(c, "owner@example.com")
    assert r.json()["user"]["id"] == 1
    assert c.get("/api/transactions").json()["total"] == seeded["transactions"]
    other = browser(app)
    signup(other, "second@example.com")
    assert other.get("/api/transactions").json()["total"] == 0


# ---------------------------------------------------------------------------- password lifecycle


def test_password_reset_flow(app, monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(
        "finpilot.api.routes.auth._send_reset", lambda request, user, token: sent.append(token)
    )
    a = browser(app)
    signup(a, "asha@example.com")

    anon = browser(app)
    r = anon.post("/api/auth/password/forgot", json={"email": "asha@example.com"})
    unknown = anon.post("/api/auth/password/forgot", json={"email": "nobody@example.com"})
    assert r.json() == unknown.json()  # does not reveal which addresses have accounts
    assert len(sent) == 1

    assert (
        anon.post(
            "/api/auth/password/reset", json={"token": "bogus", "password": "new password 1"}
        ).status_code
        == 400
    )
    r = anon.post("/api/auth/password/reset", json={"token": sent[0], "password": "new password 1"})
    assert r.status_code == 200 and anon.get("/api/auth/me").status_code == 200
    # single use, old sessions revoked, old password gone
    assert (
        anon.post(
            "/api/auth/password/reset", json={"token": sent[0], "password": "another one 2"}
        ).status_code
        == 400
    )
    assert a.get("/api/auth/me").status_code == 401
    login = browser(app).post("/api/auth/login", json={"email": "asha@example.com", "password": PASSWORD})
    assert login.status_code == 401
    login = browser(app).post(
        "/api/auth/login", json={"email": "asha@example.com", "password": "new password 1"}
    )
    assert login.status_code == 200


def test_reset_link_is_logged_without_smtp(app, caplog):
    signup(browser(app), "asha@example.com")
    with caplog.at_level("WARNING", logger="finpilot.auth"):
        browser(app).post("/api/auth/password/forgot", json={"email": "asha@example.com"})
    assert any("reset-password?token=" in r.getMessage() for r in caplog.records)


def test_change_password_keeps_this_session_and_ends_others(app):
    here, there = browser(app), browser(app)
    signup(here, "asha@example.com")
    there.post("/api/auth/login", json={"email": "asha@example.com", "password": PASSWORD})
    bad = here.post(
        "/api/auth/password/change", json={"current_password": "wrong pass", "new_password": "brand new pass"}
    )
    assert bad.status_code == 401
    ok = here.post(
        "/api/auth/password/change", json={"current_password": PASSWORD, "new_password": "brand new pass"}
    )
    assert ok.status_code == 200
    assert here.get("/api/auth/me").status_code == 200
    assert there.get("/api/auth/me").status_code == 401


def test_delete_account_removes_everything(app):
    from finpilot.db import session_scope
    from finpilot.models import Transaction, User

    c = browser(app)
    uid = signup(c, "asha@example.com").json()["user"]["id"]
    c.post("/api/transactions", json={"date": "2026-10-01", "description": "Rent", "amount": -25000})
    assert c.post("/api/auth/account/delete", json={"password": "wrong pass"}).status_code == 401
    assert c.post("/api/auth/account/delete", json={"password": PASSWORD}).status_code == 200
    assert c.get("/api/auth/me").status_code == 401
    with session_scope() as db:
        assert db.get(User, uid) is None
        assert db.query(Transaction).filter_by(user_id=uid).count() == 0


# ---------------------------------------------------------------------------- demo accounts


def test_demo_accounts_are_private_and_expire(app, monkeypatch):
    monkeypatch.setattr("finpilot.api.routes.auth._analyse", lambda uid: None)  # keep the test fast
    one, two = browser(app), browser(app)
    u1 = one.post("/api/auth/demo").json()["user"]
    u2 = two.post("/api/auth/demo").json()["user"]
    assert u1["is_demo"] and u1["id"] != u2["id"] and u1["email"] is None
    n = one.get("/api/transactions").json()["total"]
    assert n > 100
    one.post("/api/transactions", json={"date": "2026-10-01", "description": "Only mine", "amount": -1})
    assert two.get("/api/transactions").json()["total"] == n
    # demo accounts cannot touch server-wide settings
    assert one.put("/api/settings/llm", json={"enabled": False}).status_code == 403

    from finpilot.db import session_scope
    from finpilot.models import User

    with session_scope() as db:
        db.get(User, u1["id"]).created_at = dt.datetime.now(dt.UTC) - dt.timedelta(days=3)
    with session_scope() as db:
        purged = auth.purge_expired(db)
    assert purged["demo_accounts"] == 1
    assert one.get("/api/auth/me").status_code == 401
    assert two.get("/api/auth/me").status_code == 200


def test_expired_sessions_are_refused(app):
    from finpilot.db import session_scope
    from finpilot.models import AuthSession

    c = browser(app)
    signup(c, "asha@example.com")
    with session_scope() as db:
        for s in db.query(AuthSession):
            s.expires_at = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)
    assert c.get("/api/auth/me").status_code == 401


def test_single_user_mode_needs_no_login():
    with TestClient(create_app()) as c:
        me = c.get("/api/auth/me").json()
        assert me["auth_enabled"] is False and me["user"]["id"] == 1 and me["user"]["is_admin"] is True
        assert c.get("/api/dashboard").status_code == 200
        assert (
            c.post("/api/auth/login", json={"email": "a@b.co", "password": PASSWORD}, headers=XHR).status_code
            == 404
        )


def test_simultaneous_first_signups_cannot_both_claim_or_both_be_admin(accounts, seeded):
    from concurrent.futures import ThreadPoolExecutor

    from finpilot.db import session_scope

    def one(email):
        with session_scope() as db:
            u = auth.signup(db, email, PASSWORD)
            return u.id, bool(u.is_admin)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(one, [f"user{i}@example.com" for i in range(4)]))
    ids = [r[0] for r in results]
    assert len(set(ids)) == 4 and ids.count(1) == 1
    assert sum(admin for _, admin in results) == 1
