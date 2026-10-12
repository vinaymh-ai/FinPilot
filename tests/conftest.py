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
        "DATABASE_URL": f"sqlite:///{_TMP}/test.db",
        "SEED_DEMO_DATA": "false",
        "SCHEDULER_ENABLED": "false",
        "LLM_ENABLED": "false",
        "MESH_API_KEY": "",
        "LLM_FALLBACK_PROVIDERS": "",
        "MONTE_CARLO_PATHS": "200",
    }
)

import datetime as dt  # noqa: E402

import pytest  # noqa: E402

from finpilot.config import Settings, get_settings  # noqa: E402
from finpilot.db import Base, configure_database, get_engine, init_db, session_scope  # noqa: E402
from finpilot.llm.client import LLMResponse, LLMRouter, set_router  # noqa: E402
from finpilot.llm.providers import ProviderSpec  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def fresh_db() -> Iterator[None]:
    configure_database(os.environ["DATABASE_URL"])
    Base.metadata.drop_all(get_engine())
    init_db()
    set_router(LLMRouter(Settings(llm_enabled=False)))
    yield


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
