"""Test fixtures. Environment is pinned *before* finpilot is imported so settings are deterministic."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

_TMP = tempfile.mkdtemp(prefix="finpilot-tests-")
os.environ.update(
    {
        "DATABASE_URL": os.environ.get("FINPILOT_TEST_DATABASE_URL", f"sqlite:///{_TMP}/test.db"),
        "SEED_DEMO_DATA": "false",
        "SCHEDULER_ENABLED": "false",
        "LLM_ENABLED": "false",
        "LLM_PROVIDER": "mesh",
        "LLM_MODEL": "",
        "MESH_API_KEY": "",
        "FREELLMAPI_API_KEY": "",
        "TYPESAFE_API_KEY": "",
        "JEV_PROVIDER": "mesh",
        "JEV_MODEL": "typesafe/jev-1.13",
        "LLM_FALLBACK_PROVIDERS": "",
        "MONTE_CARLO_PATHS": "200",
        # single-user mode unless a test turns accounts on (see the `accounts` fixture)
        "AUTH_ENABLED": "false",
        "LOG_FILE": "",
        "LANGSMITH_TRACING": "false",
    }
)

import datetime as dt  # noqa: E402

import pytest  # noqa: E402

from finpilot.config import Settings, get_settings  # noqa: E402
from finpilot.db import Base, configure_database, get_engine, init_db, session_scope  # noqa: E402
from finpilot.decisions.jev import JevClient, set_jev  # noqa: E402
from finpilot.llm.client import LLMResponse, LLMRouter, set_router  # noqa: E402
from finpilot.llm.providers import ProviderSpec  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def fresh_db() -> Iterator[None]:
    configure_database(os.environ["DATABASE_URL"])
    Base.metadata.drop_all(get_engine())
    init_db()
    set_router(LLMRouter(Settings(llm_enabled=False)))
    set_jev(JevClient(Settings(jev_enabled=False)))
    yield


@pytest.fixture
def accounts(monkeypatch):
    """Turn on multi-user mode (e-mail + password accounts) for one test."""
    from finpilot.services.auth import limiter

    s = get_settings()
    monkeypatch.setattr(s, "auth_enabled", True)
    monkeypatch.setattr(s, "allow_signup", True)
    monkeypatch.setattr(s, "demo_accounts", True)
    monkeypatch.setattr(s, "admin_emails", "")
    limiter.reset()
    yield s
    limiter.reset()


@pytest.fixture
def db():
    with session_scope() as s:
        yield s


class FakeRouter(LLMRouter):
    """An LLM router whose 'provider' is a Python function — exercises routing, caching and telemetry."""

    def __init__(self, responder: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any] | str]):
        super().__init__(Settings(llm_enabled=True, mesh_api_key="rsk_test", llm_cache_enabled=False))
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    def _call(self, spec: ProviderSpec, model: str, messages, *, json_mode, tools, temperature, max_tokens):
        self.calls.append({"provider": spec.name, "model": model, "messages": messages, "tools": tools})
        out = self.responder(messages, {"tools": tools, "json_mode": json_mode, "model": model})
        if isinstance(out, dict) and "tool_calls" in out:
            return LLMResponse(
                content=out.get("content"),
                provider=spec.name,
                model=model,
                tool_calls=out["tool_calls"],
                prompt_tokens=10,
                completion_tokens=5,
            )
        content = out if isinstance(out, str) else json.dumps(out)
        return LLMResponse(
            content=content, provider=spec.name, model=model, prompt_tokens=10, completion_tokens=5
        )


@pytest.fixture
def fake_router_factory():
    def make(responder):
        r = FakeRouter(responder)
        set_router(r)
        return r

    return make


@pytest.fixture
def demo_end() -> dt.date:
    return dt.date(2026, 10, 12)


@pytest.fixture
def seeded(db, demo_end):
    from finpilot.services.seed import seed_demo

    info = seed_demo(db, end=demo_end)
    db.commit()
    return info


# ----------------------------------------------------------------------------- simulated System One

Picker = Callable[[Any, list[str]], "dict[str, float] | str | None"]


class FakeJev:
    """A stand-in for Mesh `POST /v1/evaluate`, served through httpx.MockTransport.

    Each question name can have a picker: (state, option_keys) → preferred key (answered with
    p=0.92) or a full probability dict. Unknown questions get a flat distribution (confidence 0),
    which exercises the "defer" paths. Every request body is recorded for assertions.
    """

    def __init__(self, pickers: dict[str, Picker] | None = None, *, nouls: dict[str, float] | None = None):
        import httpx

        self.pickers = dict(pickers or {})
        self.nouls = dict(nouls or {})
        self.requests: list[dict[str, Any]] = []
        self.fail_next: list[int] = []  # status codes to return before succeeding
        self.transport = httpx.MockTransport(self._handle)

    def client(self, **overrides: Any) -> JevClient:
        settings = Settings(mesh_api_key="rsk_test", jev_enabled=True, **overrides)
        return JevClient(settings, transport=self.transport)

    def _handle(self, request):
        import httpx

        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), text="simulated upstream error")
        body = json.loads(request.content)
        self.requests.append(body)
        answers: dict[str, Any] = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": self.nouls.get(name, 0.5)}
                continue
            keys = [
                str(k)
                for k in (q["criteria"] if isinstance(q["criteria"], dict) else range(len(q["criteria"])))
            ]
            pick = self.pickers.get(name)
            out = pick(body["state"], keys) if pick else None
            if isinstance(out, str) and out in keys:
                probs = {k: (0.92 if k == out else 0.08 / (len(keys) - 1)) for k in keys}
            elif isinstance(out, dict):
                probs = {k: float(out.get(k, 0.0)) for k in keys}
                total = sum(probs.values()) or 1.0
                probs = {k: v / total for k, v in probs.items()}
            else:
                probs = {k: 1 / len(keys) for k in keys}
            top = max(probs, key=lambda k: probs[k])
            n = len(keys)
            conf = (probs[top] - 1 / n) / (1 - 1 / n)
            if q["type"] == "choice":
                answers[name] = {"type": "choice", "choice": top, "probabilities": probs, "confidence": conf}
            else:
                score = sum(int(k) * v for k, v in probs.items())
                legend = {str(i): c for i, c in enumerate(q["criteria"])}
                answers[name] = {
                    "type": "score",
                    "score": score,
                    "legend": legend,
                    "probabilities": probs,
                    "confidence": conf,
                }
        return httpx.Response(
            200,
            json={
                "id": f"eval_{len(self.requests)}",
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": len(request.content) // 4, "output_tokens": 20},
            },
        )


@pytest.fixture
def fake_jev_factory():
    def make(pickers: dict[str, Picker] | None = None, *, nouls: dict[str, float] | None = None, **overrides):
        fake = FakeJev(pickers, nouls=nouls)
        client = fake.client(**overrides)
        set_jev(client)
        return fake, client

    return make
