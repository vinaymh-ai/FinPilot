"""Observability: request ids, raw model-call traces, structured log file and viewer, client errors."""

from __future__ import annotations

import datetime as dt
import json
import logging

import pytest
from fastapi.testclient import TestClient

from finpilot.config import get_settings
from finpilot.db import session_scope
from finpilot.logging_config import configure_logging
from finpilot.main import create_app
from finpilot.models import LLMCall
from finpilot.observability import clip, configure_langsmith, user_context
from finpilot.observability.context import new_request_id

XHR = {"X-Requested-With": "FinPilot"}


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def log_file(tmp_path):
    """Route logs to a JSON-lines file for one test, then restore the default handlers."""
    path = tmp_path / "finpilot.log"
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    s = get_settings()
    old = s.log_file
    s.log_file = str(path)
    configure_logging("DEBUG", False, str(path))
    yield path
    s.log_file = old
    for h in root.handlers:
        h.close()
    root.handlers[:], root.level = saved


def _lines(path):
    for h in logging.getLogger().handlers:
        h.flush()
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------------------- request context


def test_every_response_carries_a_request_id_and_security_headers(client):
    r = client.get("/api/health")
    assert len(r.headers["x-request-id"]) >= 8
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    # a well-formed upstream id (from a proxy) is kept, so logs can be joined across services
    assert (
        client.get("/api/health", headers={"X-Request-ID": "edge-12345678"}).headers["x-request-id"]
        == "edge-12345678"
    )
    assert client.get("/api/health", headers={"X-Request-ID": "bad id!"}).headers["x-request-id"] != "bad id!"


def test_request_id_validation():
    assert new_request_id("abcDEF12-._") == "abcDEF12-._"
    assert new_request_id("short") != "short"
    assert len(new_request_id(None)) == 16


def test_unhandled_errors_return_the_request_id(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("finpilot.api.routes.core.repo.get_or_create_user", boom)
    c = TestClient(client.app, raise_server_exceptions=False)
    r = c.get("/api/meta")
    assert r.status_code == 500
    assert r.json()["detail"]["request_id"]
    assert "kaboom" not in r.text  # internals stay in the log


# ---------------------------------------------------------------------------- model-call traces


def test_llm_calls_keep_raw_request_and_response(client, seeded, fake_router_factory):
    fake_router_factory(lambda messages, opts: "You spent ₹1,234 on dining.")
    r = client.post(
        "/api/chat",
        json={"message": "write me a haiku about my spending"},
        headers={"X-Request-ID": "trace-test-0001"},
    )
    assert r.status_code == 200
    calls = client.get("/api/observability/calls", params={"kind": "llm"}).json()["calls"]
    assert calls and calls[0]["has_payload"] and calls[0]["user_id"] == 1
    assert calls[0]["request_id"] == "trace-test-0001"
    detail = client.get(f"/api/observability/calls/{calls[0]['id']}").json()["call"]
    req = detail["request"]
    assert req["provider"] == "mesh" and req["messages"][-1]["content"].startswith("write me a haiku")
    assert "rsk_" not in json.dumps(req)  # keys are never part of a trace
    assert detail["response"]["content"] == "You spent ₹1,234 on dining."
    # filtering by request id finds every call that request made
    by_req = client.get("/api/observability/calls", params={"request_id": "trace-test-0001"}).json()["calls"]
    assert {c["id"] for c in by_req} >= {calls[0]["id"]}


def test_failed_calls_are_traced_with_the_error(seeded, fake_router_factory):
    from finpilot.llm.client import LLMError, get_router

    def broken(messages, opts):
        raise TimeoutError("upstream timed out")

    fake_router_factory(broken)
    with user_context(1), pytest.raises(LLMError):
        get_router().chat([{"role": "user", "content": "hi"}], agent="chat", cache=False)
    with session_scope() as db:
        c = db.query(LLMCall).order_by(LLMCall.id.desc()).first()
        assert c.success is False and "upstream timed out" in c.error
        assert c.request_payload["messages"][0]["content"] == "hi" and c.response_payload is None
        assert c.user_id == 1


def test_jev_calls_keep_raw_payloads(fake_jev_factory):
    from finpilot.decisions.jev import Choice

    fake, client = fake_jev_factory({"pick": lambda state, keys: "b"})
    with user_context(7):
        client.evaluate(
            {"x": 1}, {"pick": Choice("Which?", {"a": "A", "b": "B"})}, agent="test", purpose="unit"
        )
    with session_scope() as db:
        c = db.query(LLMCall).filter_by(kind="jev").one()
        assert c.user_id == 7 and c.purpose == "unit"
        assert c.request_payload["state"] == {"x": 1} and "pick" in c.request_payload["questions"]
        assert c.response_payload["answers"]["pick"]["choice"] == "b"


def test_payload_capture_can_be_turned_off(fake_router_factory, monkeypatch):
    from finpilot.llm.client import get_router

    monkeypatch.setattr(get_settings(), "trace_payloads", False)
    fake_router_factory(lambda m, o: "ok")
    get_router().chat([{"role": "user", "content": "secret"}], agent="chat", cache=False)
    with session_scope() as db:
        c = db.query(LLMCall).one()
        assert c.request_payload is None and c.response_payload is None and c.success


def test_clip_truncates_large_payloads():
    out = clip({"a": "x" * 50, "b": list(range(300)), "c": object()}, 10)
    assert out["a"].startswith("x" * 10) and "+40 chars" in out["a"]
    assert len(out["b"]) == 201 and "+100 items" in out["b"][-1]
    assert isinstance(out["c"], str)


def test_calls_are_private_to_their_user(accounts):
    app = create_app()
    a, b = TestClient(app, headers=XHR), TestClient(app, headers=XHR)
    a.post("/api/auth/signup", json={"email": "asha@example.com", "password": "correct horse 1"})
    b.post("/api/auth/signup", json={"email": "ravi@example.com", "password": "correct horse 2"})
    with session_scope() as db:
        call = LLMCall(
            agent="chat", provider="mesh", model="m", user_id=1, success=True, request_payload={"q": 1}
        )
        db.add(call)
        db.flush()
        cid = call.id
    assert a.get(f"/api/observability/calls/{cid}").status_code == 200
    assert b.get(f"/api/observability/calls/{cid}").status_code == 404
    assert b.get("/api/observability/calls").json()["calls"] == []
    assert b.get("/api/observability/calls", params={"all_users": True}).json()["calls"] == []  # not admin
    assert len(a.get("/api/observability/calls", params={"all_users": True}).json()["calls"]) == 1


def test_summary(client, seeded, fake_router_factory):
    fake_router_factory(lambda m, o: "fine")
    client.post("/api/chat", json={"message": "write a poem about rent"})
    s = client.get("/api/observability/summary").json()
    assert s["by_kind"]["llm"]["calls"] >= 1 and s["latency_ms"]["p50"] is not None


def test_maintenance_purges_old_payloads():
    from finpilot.services.scheduler import run_maintenance

    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=60)
    with session_scope() as db:
        db.add(
            LLMCall(
                agent="chat", provider="p", model="m", success=True, request_payload={"q": 1}, created_at=old
            )
        )
        db.add(LLMCall(agent="chat", provider="p", model="m", success=True, request_payload={"q": 2}))
    assert run_maintenance()["payloads_purged"] == 1
    with session_scope() as db:
        kept = [c.request_payload for c in db.query(LLMCall).order_by(LLMCall.id)]
    assert kept == [None, {"q": 2}]


# ---------------------------------------------------------------------------- logs


def test_log_file_is_structured_and_searchable(client, log_file):
    client.get("/api/dashboard", headers={"X-Request-ID": "log-test-00001"})
    with user_context(1):
        logging.getLogger("finpilot.test").info("hello %s", "world", extra={"event": "unit", "answer": 42})
    records = _lines(log_file)
    access = [
        r for r in records if r.get("event") == "http_request" and r.get("request_id") == "log-test-00001"
    ]
    assert access and access[0]["status"] == 200 and access[0]["path"] == "/api/dashboard"
    assert access[0]["user_id"] == 1 and access[0]["duration_ms"] >= 0
    unit = next(r for r in records if r.get("event") == "unit")
    assert unit["msg"] == "hello world" and unit["answer"] == 42 and unit["user_id"] == 1

    view = client.get("/api/observability/logs", params={"request_id": "log-test-00001"}).json()
    assert (
        view["enabled"] and view["lines"] and all(x["request_id"] == "log-test-00001" for x in view["lines"])
    )
    view = client.get("/api/observability/logs", params={"q": "hello world"}).json()
    assert [x["msg"] for x in view["lines"]] == ["hello world"]
    assert (
        client.get("/api/observability/logs", params={"level": "ERROR", "q": "hello"}).json()["lines"] == []
    )


def test_client_errors_are_logged(client, log_file):
    r = client.post(
        "/api/client-errors",
        json={
            "event_id": "ab12cd34",
            "message": "Failed to fetch dynamically imported module",
            "url": "http://x/chat",
            "context": {"page": "chat"},
        },
    )
    assert r.status_code == 202 and r.json()["accepted"]
    rec = next(x for x in _lines(log_file) if x.get("event") == "client_error")
    assert rec["level"] == "ERROR" and rec["event_id"] == "ab12cd34" and rec["context"] == {"page": "chat"}


def test_status_endpoint(client):
    s = client.get("/api/observability/status").json()
    assert s["langsmith"]["enabled"] is False and s["trace_payloads"] is True


def test_langsmith_needs_a_key(monkeypatch):
    s = get_settings().model_copy(update={"langsmith_tracing": True, "langsmith_api_key": None})
    assert configure_langsmith(s)["enabled"] is False
    s = get_settings().model_copy(
        update={"langsmith_tracing": True, "langsmith_api_key": "lsv2_test", "langsmith_project": "fp"}
    )
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    try:
        st = configure_langsmith(s)
        import os

        assert st["enabled"] and os.environ["LANGSMITH_PROJECT"] == "fp"
    finally:
        configure_langsmith(get_settings())
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)


# ---------------------------------------------------------------------------- eval preflight


def test_jev_eval_preflight_explains_a_rejected_key(fake_jev_factory):
    from evaluation.jev_eval import JevEvalError, pick_threshold, preflight

    fake, client = fake_jev_factory(jev_max_retries=0)
    fake.fail_next = [401]
    with pytest.raises(JevEvalError) as err:
        preflight(client)
    assert "rejected the key" in str(err.value) and "FreeLLMAPI" in str(err.value)
    assert pick_threshold([], 0.9) is None
    assert pick_threshold([{"threshold": 0.5, "accuracy": None, "coverage": 0.0}], 0.9) is None
