"""Provider-agnostic LLM router.

Responsibilities:
* resolve which provider/model each agent should use (env defaults → runtime overrides),
* call it through the OpenAI SDK pointed at that provider's base URL (Mesh by default),
* fall back to the next configured provider on failure,
* cache deterministic calls (saves money — a priority when running on free/low budgets),
* record every call (tokens, latency, success, raw request/response) against the active agent run,
  user and HTTP request for observability (see finpilot.observability).

If no provider is usable the router reports `available() == False` and every agent degrades
to its deterministic path, so the product keeps working with zero LLM spend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from openai import APIStatusError, BadRequestError, OpenAI

from finpilot.config import Settings, get_settings
from finpilot.llm.providers import (
    PROVIDERS,
    ProviderSpec,
    is_configured,
    resolve_api_key,
    resolve_base_url,
)
from finpilot.observability.calls import record_model_call
from finpilot.observability.tracing import wrap_openai_client

log = logging.getLogger(__name__)

AGENTS = ("categorizer", "analyzer", "forecaster", "advisor", "chat", "digest")

_run_id: ContextVar[int | None] = ContextVar("finpilot_run_id", default=None)
_step_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("finpilot_step_calls", default=None)


@contextmanager
def run_context(run_id: int | None) -> Iterator[None]:
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


@contextmanager
def step_call_log() -> Iterator[list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    token = _step_calls.set(calls)
    try:
        yield calls
    finally:
        _step_calls.reset(token)


def current_run_id() -> int | None:
    return _run_id.get()


def log_step_call(entry: dict[str, Any]) -> None:
    """Attach a model call (LLM or Jev) to the agent step that is currently running, if any."""
    calls = _step_calls.get()
    if calls is not None:
        calls.append(entry)


def llm_cost_usd(prompt_tokens: int, completion_tokens: int, settings: Settings | None = None) -> float:
    s = settings or get_settings()
    return (prompt_tokens * s.llm_price_in_per_1m + completion_tokens * s.llm_price_out_per_1m) / 1e6


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    pass


@dataclass
class LLMResponse:
    content: str | None
    provider: str
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False
    raw: dict[str, Any] | None = field(default=None, repr=False)  # provider's response, for traces

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def assistant_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in self.tool_calls
            ]
        return msg


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and leading/trailing prose."""
    if text is None:
        raise ValueError("empty response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(cleaned[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model response")


class LLMRouter:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._clients: dict[tuple[str, str, str], OpenAI] = {}
        self._lock = threading.Lock()
        # Runtime overrides (set from the Settings page, persisted in app_settings):
        # {"enabled": bool, "provider": str, "model": str, "agents": {agent: {"provider","model"}}}
        self._overrides: dict[str, Any] = {}

    # ----------------------------------------------------------------------------- config
    def set_overrides(self, overrides: dict[str, Any] | None) -> None:
        self._overrides = dict(overrides or {})

    @property
    def overrides(self) -> dict[str, Any]:
        return dict(self._overrides)

    @property
    def enabled(self) -> bool:
        return bool(self._overrides.get("enabled", self.settings.llm_enabled))

    def _agent_default_model(self, agent: str) -> str | None:
        return getattr(self.settings, f"{agent}_model", None)

    def targets(self, agent: str) -> list[tuple[ProviderSpec, str]]:
        """Ordered (provider, model) candidates for an agent: primary first, then fallbacks."""
        agent_ovr = (self._overrides.get("agents") or {}).get(agent) or {}
        env_provider = self.settings.llm_provider
        if agent_ovr.get("provider"):
            primary_name = agent_ovr["provider"]
            candidates = [agent_ovr.get("model")]
        else:
            primary_name = self._overrides.get("provider") or env_provider
            candidates = [agent_ovr.get("model"), self._overrides.get("model")]
        if primary_name == env_provider:
            # env-level defaults only make sense for the provider they were written for
            candidates += [self._agent_default_model(agent), self.settings.llm_model]
        if primary_name == "custom":
            candidates.append(self.settings.custom_llm_model)
        primary = PROVIDERS.get(primary_name)
        out: list[tuple[ProviderSpec, str]] = []
        if primary is not None:
            model = next((c for c in candidates if c), primary.default_model)
            out.append((primary, model))
        for name in self.settings.fallback_provider_list:
            spec = PROVIDERS.get(name)
            if spec and spec.name != primary_name:
                model = self.settings.custom_llm_model if name == "custom" else None
                out.append((spec, model or spec.default_model))
        return [(s, m) for s, m in out if is_configured(s, self.settings)]

    def available(self, agent: str = "advisor") -> bool:
        return self.enabled and bool(self.targets(agent))

    def describe(self, agent: str) -> str:
        t = self.targets(agent) if self.enabled else []
        return f"{t[0][0].name}:{t[0][1]}" if t else "rules"

    def _client(self, spec: ProviderSpec) -> OpenAI:
        base_url = resolve_base_url(spec, self.settings)
        key = resolve_api_key(spec, self.settings) or ""
        cache_key = (spec.name, base_url, hashlib.sha256(key.encode()).hexdigest()[:12])
        with self._lock:
            client = self._clients.get(cache_key)
            if client is None:
                client = wrap_openai_client(
                    OpenAI(
                        base_url=base_url,
                        api_key=key,
                        timeout=self.settings.llm_timeout_s,
                        max_retries=self.settings.llm_max_retries,
                        default_headers=spec.extra_headers or None,
                    )
                )
                self._clients[cache_key] = client
        return client

    # ----------------------------------------------------------------------------- calls
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        agent: str,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int = 1200,
        cache: bool = True,
        model: str | None = None,
    ) -> LLMResponse:
        if not self.enabled:
            raise LLMUnavailableError("LLM disabled")
        targets = self.targets(agent)
        if not targets:
            raise LLMUnavailableError("no LLM provider configured")
        if model:  # per-call override on the primary provider (e.g. a cheaper model for simple questions)
            targets = [(targets[0][0], model)] + targets[1:]
        temp = self.settings.llm_temperature if temperature is None else temperature
        use_cache = cache and self.settings.llm_cache_enabled and not tools and temp <= 0.3

        errors: list[str] = []
        for spec, model in targets:
            request = {
                "provider": spec.name,
                "model": model,
                "messages": messages,
                "tools": [t.get("function", {}).get("name") for t in tools] if tools else None,
                "json_mode": json_mode,
                "temperature": temp,
                "max_tokens": max_tokens,
            }
            cache_key = self._cache_key(spec.name, model, messages, json_mode, temp) if use_cache else None
            if cache_key:
                hit = self._cache_get(cache_key)
                if hit is not None:
                    resp = LLMResponse(content=hit, provider=spec.name, model=model, cached=True)
                    self._record(agent, resp, success=True, request=request)
                    return resp
            try:
                resp = self._call(
                    spec,
                    model,
                    messages,
                    json_mode=json_mode,
                    tools=tools,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - provider errors are heterogeneous
                msg = f"{spec.name}:{model}: {type(exc).__name__}: {str(exc)[:300]}"
                errors.append(msg)
                log.warning("LLM call failed (%s) — trying next provider", msg)
                self._record(
                    agent,
                    LLMResponse(content=None, provider=spec.name, model=model),
                    success=False,
                    error=msg,
                    request=request,
                )
                continue
            if cache_key and resp.content:
                self._cache_put(cache_key, model, resp.content)
            self._record(agent, resp, success=True, request=request)
            return resp
        raise LLMError("; ".join(errors) or "all providers failed")

    def chat_json(
        self, messages: list[dict[str, Any]], *, agent: str, **kwargs: Any
    ) -> tuple[Any, LLMResponse]:
        resp = self.chat(messages, agent=agent, json_mode=True, **kwargs)
        return extract_json(resp.content or ""), resp

    def _call(
        self,
        spec: ProviderSpec,
        model: str,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        client = self._client(spec)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode and spec.supports_json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            # Some models reject response_format; retry once without it (we parse leniently anyway).
            if "response_format" in kwargs and "response_format" in str(exc).lower():
                kwargs.pop("response_format")
                completion = client.chat.completions.create(**kwargs)
            else:
                raise
        except APIStatusError:
            raise
        latency = int((time.perf_counter() - started) * 1000)
        choice = completion.choices[0]
        tool_calls = []
        for tc in choice.message.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tool_calls.append({"id": tc.id, "name": fn.name, "arguments": fn.arguments or "{}"})
        usage = completion.usage
        return LLMResponse(
            content=choice.message.content,
            provider=spec.name,
            model=getattr(completion, "model", None) or model,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency,
            raw=_dump(completion),
        )

    # ----------------------------------------------------------------------------- cache/telemetry
    @staticmethod
    def _cache_key(
        provider: str, model: str, messages: list[dict[str, Any]], json_mode: bool, temp: float
    ) -> str:
        blob = json.dumps(
            {"p": provider, "m": model, "msgs": messages, "j": json_mode, "t": temp},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        from finpilot.db import session_scope
        from finpilot.models import LLMCache

        try:
            with session_scope() as s:
                row = s.get(LLMCache, key)
                return row.response if row else None
        except Exception:  # noqa: BLE001
            return None

    def _cache_put(self, key: str, model: str, response: str) -> None:
        from finpilot.db import session_scope
        from finpilot.models import LLMCache

        try:
            with session_scope() as s:
                s.merge(LLMCache(key=key, model=model, response=response))
        except Exception:  # noqa: BLE001
            log.debug("cache write failed", exc_info=True)

    def _record(
        self,
        agent: str,
        resp: LLMResponse,
        *,
        success: bool,
        error: str | None = None,
        request: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "agent": agent,
            "provider": resp.provider,
            "model": resp.model,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_ms": resp.latency_ms,
            "cached": resp.cached,
            "success": success,
            "kind": "llm",
            "cost_usd": round(llm_cost_usd(resp.prompt_tokens, resp.completion_tokens, self.settings), 8),
        }
        log_step_call(entry)
        response = None
        if success:
            response = resp.raw or {
                "content": resp.content,
                "tool_calls": resp.tool_calls or None,
                "cached": resp.cached,
            }
        record_model_call(entry, run_id=_run_id.get(), error=error, request=request, response=response)


def _dump(completion: Any) -> dict[str, Any] | None:
    try:
        return completion.model_dump(exclude_none=True)
    except Exception:  # noqa: BLE001 - a stub or non-pydantic response
        return None


_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


def set_router(router: LLMRouter) -> None:
    """Swap the global router (used by tests to inject a fake model)."""
    global _router
    _router = router
