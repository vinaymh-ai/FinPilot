"""System One client — typed, calibrated decisions from TypeSafe's Jev.

An LLM *generates*; Jev *decides*. You send application `state` plus named questions whose
answers come from a fixed set, and get back a probability for every option instead of prose:

* `Choice` — pick one of your options (up to 255) → winning key + probability per option
* `Score`  — rate on an ordered scale (lowest first) → expected position + distribution
* `Noul`   — a single probability in [0, 1] (not a boolean; threshold it yourself)

Default route is Mesh API `POST /v1/evaluate` (model `typesafe/jev-1.13`), so every AI call in
FinPilot still goes through Mesh. `JEV_PROVIDER=typesafe` calls TypeSafe directly instead.

What this client adds on top of the HTTP call:
* client-side validation that mirrors the API's 422 rules (a bad question never costs a request),
* bounded retries on 429/503/504 (Mesh documents 500 as "do not retry"),
* a response cache keyed on (model, state, questions) — repeated decisions are free,
* per-call telemetry (tokens, latency, USD) into the same `llm_calls` table as the LLM router,
  tagged `kind="jev"` and attached to the running agent step,
* `evaluate_many` — fan-out across states in parallel, preserving the run/step context.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

from finpilot.config import Settings, get_settings
from finpilot.llm.client import current_run_id, log_step_call
from finpilot.observability.calls import record_model_call
from finpilot.observability.tracing import traceable

log = logging.getLogger(__name__)

QUESTION_KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_CHOICE_OPTIONS = 255
RETRY_STATUS = {429, 503, 504}


class JevError(RuntimeError):
    """The decision could not be made (network, provider or contract error)."""


class JevUnavailableError(JevError):
    """Jev is disabled or has no credentials — callers use their deterministic path."""


class JevRequestError(JevError, ValueError):
    """The request is invalid (client-side validation or a 4xx from the API)."""


# ====================================================================== questions

Criterion = str | dict[str, Any] | list[Any]


@dataclass(frozen=True)
class Choice:
    """Pick one option. Criteria map option key → description (a string, or a structured object
    such as {"what": ..., "not_for": ..., "examples": [...]} for options that get confused)."""

    instructions: str
    criteria: dict[str, Criterion]
    type: ClassVar[str] = "choice"

    def reversed(self) -> Choice:
        """Same question, options in reverse order — used to detect option-order bias."""
        return Choice(self.instructions, dict(reversed(list(self.criteria.items()))))


@dataclass(frozen=True)
class Score:
    """Rate on an ordered scale. `criteria` is lowest first; its order is part of the schema."""

    instructions: str
    criteria: list[str]
    type: ClassVar[str] = "score"


@dataclass(frozen=True)
class Noul:
    """A probability in [0, 1]. Despite the name it is not a boolean."""

    instructions: str
    type: ClassVar[str] = "noul"


Question = Choice | Score | Noul


def validate_questions(questions: dict[str, Question]) -> None:
    """Mirror of the API's request rules, so a malformed question fails before any network call."""
    if not questions:
        raise JevRequestError("at least one question is required")
    for name, q in questions.items():
        if not QUESTION_KEY.match(name):
            raise JevRequestError(f"question key {name!r} must match {QUESTION_KEY.pattern}")
        if not isinstance(q.instructions, str) or not q.instructions.strip():
            raise JevRequestError(f"{name}: instructions cannot be blank")
        if isinstance(q, Choice):
            if len(q.criteria) < 2:
                raise JevRequestError(f"{name}: a choice needs at least two options")
            if len(q.criteria) > MAX_CHOICE_OPTIONS:
                raise JevRequestError(f"{name}: at most {MAX_CHOICE_OPTIONS} options")
            for key, desc in q.criteria.items():
                if not str(key).strip() or not _render_criterion(desc, structured=False).strip():
                    raise JevRequestError(f"{name}: blank option or description")
        elif isinstance(q, Score):
            if len(q.criteria) < 2 or any(not str(c).strip() for c in q.criteria):
                raise JevRequestError(f"{name}: a score needs at least two non-blank scale points")
        elif not isinstance(q, Noul):
            raise JevRequestError(f"{name}: unknown question type {type(q).__name__}")


def _render_criterion(value: Criterion, *, structured: bool) -> Any:
    """Structured criteria go as-is to TypeSafe; Mesh documents string descriptions, so flatten."""
    if structured or isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            text = "; ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
            if text:
                parts.append(text if k in ("what", "description") else f"{k.replace('_', ' ')}: {text}")
        return ". ".join(parts)
    return "; ".join(map(str, value))


def question_wire(q: Question, *, structured: bool) -> dict[str, Any]:
    body: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
    if isinstance(q, Choice):
        body["criteria"] = {k: _render_criterion(v, structured=structured) for k, v in q.criteria.items()}
    elif isinstance(q, Score):
        body["criteria"] = list(q.criteria)
    return body


# ====================================================================== answers


def confidence_from(probabilities: dict[str, float]) -> float:
    """TypeSafe's confidence: how far the top probability sits above a uniform guess, in [0, 1]."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    pmax = max(probabilities.values())
    return max(0.0, min(1.0, (pmax - 1 / n) / (1 - 1 / n)))


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: str = "choice"

    def p(self, option: str) -> float:
        return float(self.probabilities.get(option, 0.0))

    @property
    def runner_up(self) -> tuple[str, float] | None:
        rest = sorted(
            ((k, v) for k, v in self.probabilities.items() if k != self.choice), key=lambda kv: -kv[1]
        )
        return rest[0] if rest else None

    @property
    def value(self) -> str:
        return self.choice


@dataclass
class ScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float
    type: str = "score"

    @property
    def level(self) -> int:
        """Most likely scale position (0 = first/lowest entry you sent)."""
        if self.probabilities:
            return int(max(self.probabilities.items(), key=lambda kv: kv[1])[0])
        return round(self.score)

    @property
    def label(self) -> str:
        return self.legend.get(str(self.level), str(self.level))

    @property
    def value(self) -> str:
        return f"{self.score:.2f}"


@dataclass
class NoulAnswer:
    noul: float
    type: str = "noul"

    @property
    def confidence(self) -> float:
        """Distance from a coin flip: 0.5 → 0, 0 or 1 → 1."""
        return abs(2 * self.noul - 1)

    @property
    def probabilities(self) -> dict[str, float]:
        return {"yes": round(self.noul, 4), "no": round(1 - self.noul, 4)}

    @property
    def value(self) -> str:
        return f"{self.noul:.2f}"


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass
class Evaluation:
    answers: dict[str, Answer]
    model: str
    provider: str
    input_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False
    request_id: str | None = None
    cost_usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Any:
        return self.answers[name]

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


# ====================================================================== client


class JevClient:
    def __init__(self, settings: Settings | None = None, *, transport: httpx.BaseTransport | None = None):
        self.settings = settings or get_settings()
        self._transport = transport
        self._http: httpx.Client | None = None
        self._lock = threading.Lock()
        # Runtime overrides from the Settings page: {"enabled", "accept_confidence", "route_confidence",
        # "guard_confidence", "escalate_to_llm", "order_check"}
        self._overrides: dict[str, Any] = {}

    # ----------------------------------------------------------------------------- config
    def set_overrides(self, overrides: dict[str, Any] | None) -> None:
        self._overrides = dict(overrides or {})

    @property
    def overrides(self) -> dict[str, Any]:
        return dict(self._overrides)

    def _opt(self, key: str, default: Any) -> Any:
        v = self._overrides.get(key)
        return default if v is None else v

    @property
    def enabled(self) -> bool:
        return bool(self._opt("enabled", self.settings.jev_enabled))

    @property
    def accept_confidence(self) -> float:
        return float(self._opt("accept_confidence", self.settings.jev_accept_confidence))

    @property
    def route_confidence(self) -> float:
        return float(self._opt("route_confidence", self.settings.jev_route_confidence))

    @property
    def guard_confidence(self) -> float:
        return float(self._opt("guard_confidence", self.settings.jev_guard_confidence))

    @property
    def escalate_to_llm(self) -> bool:
        return bool(self._opt("escalate_to_llm", self.settings.categorizer_escalate_to_llm))

    @property
    def order_check(self) -> bool:
        return bool(self._opt("order_check", self.settings.jev_order_check))

    @property
    def provider(self) -> str:
        return "typesafe" if self.settings.jev_provider == "typesafe" else "mesh"

    @property
    def model(self) -> str:
        m = self.settings.jev_model
        # Mesh namespaces models by vendor ("typesafe/jev-1.13"); TypeSafe's own API does not.
        return m.split("/", 1)[1] if self.provider == "typesafe" and m.startswith("typesafe/") else m

    def _endpoint(self) -> tuple[str, str | None]:
        if self.provider == "typesafe":
            return f"{self.settings.typesafe_base_url.rstrip('/')}/systemone", self.settings.typesafe_api_key
        return f"{self.settings.mesh_base_url.rstrip('/')}/evaluate", self.settings.mesh_api_key

    @property
    def configured(self) -> bool:
        return bool((self._endpoint()[1] or "").strip())

    def available(self) -> bool:
        return self.enabled and self.configured

    def describe(self) -> str:
        return f"{self.provider}:{self.model}" if self.available() else "off"

    def cost_usd(self, input_tokens: int) -> float:
        return input_tokens * self.settings.jev_price_in_per_1m / 1e6

    def _client(self) -> httpx.Client:
        with self._lock:
            if self._http is None:
                self._http = httpx.Client(timeout=self.settings.jev_timeout_s, transport=self._transport)
            return self._http

    # ----------------------------------------------------------------------------- calls
    def body(
        self, state: Any, questions: dict[str, Question], *, agent: str = "", purpose: str = ""
    ) -> dict[str, Any]:
        structured = self.provider == "typesafe"
        body: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {n: question_wire(q, structured=structured) for n, q in questions.items()},
        }
        if self.provider == "mesh":
            # Attribution only (never forwarded to the model): spend per agent/purpose in Mesh usage reports.
            body["tags"] = {
                k: v for k, v in {"app": "finpilot", "agent": agent, "purpose": purpose}.items() if v
            }
        return body

    @traceable("jev.evaluate", run_type="llm")
    def evaluate(
        self,
        state: Any,
        questions: dict[str, Question],
        *,
        agent: str,
        purpose: str,
        cache: bool = True,
    ) -> Evaluation:
        if not self.available():
            raise JevUnavailableError("Jev disabled" if not self.enabled else "no Jev credentials configured")
        validate_questions(questions)
        body = self.body(state, questions, agent=agent, purpose=purpose)
        use_cache = cache and self.settings.jev_cache_enabled
        key = self._cache_key(body) if use_cache else None
        if key:
            hit = self._cache_get(key)
            if hit is not None:
                try:
                    ev = self._parse(hit, questions, latency_ms=0)
                    ev.cached, ev.input_tokens, ev.cost_usd = True, 0, 0.0
                    self._record(agent, purpose, ev, success=True, request=body, response=hit)
                    return ev
                except JevError:
                    pass  # stale/incompatible cache entry → ask again
        data: dict[str, Any] | None = None
        try:
            data, latency = self._post(body)
            ev = self._parse(data, questions, latency_ms=latency)
        except JevError as exc:
            self._record(
                agent, purpose, None, success=False, error=str(exc)[:500], request=body, response=data
            )
            raise
        if key:
            self._cache_put(key, data)
        self._record(agent, purpose, ev, success=True, request=body, response=data)
        return ev

    def evaluate_many(
        self,
        jobs: Sequence[tuple[Any, dict[str, Question]]],
        *,
        agent: str,
        purpose: str,
        cache: bool = True,
    ) -> list[Evaluation | JevError]:
        """Evaluate many independent states concurrently. Never raises: failures come back as JevError.

        Each worker runs in a copy of the caller's context, so the calls land on the current agent
        run and step exactly as if they had been made sequentially.
        """

        def one(state: Any, qs: dict[str, Question]) -> Evaluation | JevError:
            try:
                return self.evaluate(state, qs, agent=agent, purpose=purpose, cache=cache)
            except JevError as exc:
                return exc

        if len(jobs) <= 1:
            return [one(s, q) for s, q in jobs]
        workers = max(1, min(self.settings.jev_max_concurrency, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jev") as pool:
            futures = [pool.submit(contextvars.copy_context().run, one, s, q) for s, q in jobs]
            return [f.result() for f in futures]

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        url, key = self._endpoint()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if self.provider == "mesh":
            headers["X-Mesh-Version"] = self.settings.mesh_api_version
        attempts = max(1, self.settings.jev_max_retries + 1)
        last: str = ""
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                resp = self._client().post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < attempts:
                    time.sleep(0.4 * 2**attempt)
                    continue
                raise JevError(f"network error calling {self.provider}: {last}") from exc
            latency = int((time.perf_counter() - started) * 1000)
            if resp.status_code == 200:
                try:
                    return resp.json(), latency
                except ValueError as exc:
                    raise JevError("response was not JSON") from exc
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in RETRY_STATUS and attempt + 1 < attempts:
                time.sleep(_retry_after(resp) or 0.4 * 2**attempt)
                continue
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise JevRequestError(last)
            raise JevError(last)
        raise JevError(last or "evaluation failed")

    def _parse(self, data: dict[str, Any], questions: dict[str, Question], *, latency_ms: int) -> Evaluation:
        raw = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            raise JevError("response has no answers")
        answers: dict[str, Answer] = {}
        for name, q in questions.items():
            a = raw.get(name)
            if not isinstance(a, dict):
                raise JevError(f"no answer for {name!r}")
            try:
                if isinstance(q, Choice):
                    probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
                    choice = str(a.get("choice") or (max(probs, key=probs.get) if probs else ""))
                    if choice not in q.criteria:
                        raise JevError(f"{name}: answer {choice!r} is not one of the options")
                    conf = a.get("confidence")
                    answers[name] = ChoiceAnswer(
                        choice, probs, float(conf) if conf is not None else confidence_from(probs)
                    )
                elif isinstance(q, Score):
                    probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
                    legend = {str(k): str(v) for k, v in (a.get("legend") or {}).items()} or {
                        str(i): c for i, c in enumerate(q.criteria)
                    }
                    conf = a.get("confidence")
                    answers[name] = ScoreAnswer(
                        float(a["score"]),
                        legend,
                        probs,
                        float(conf) if conf is not None else confidence_from(probs),
                    )
                else:
                    answers[name] = NoulAnswer(min(1.0, max(0.0, float(a["noul"]))))
            except (KeyError, TypeError, ValueError) as exc:
                raise JevError(f"{name}: malformed answer ({exc})") from exc
        usage = data.get("usage") or {}
        tokens = int(usage.get("input_tokens") or 0)
        return Evaluation(
            answers=answers,
            model=str(data.get("model") or self.model),
            provider=self.provider,
            input_tokens=tokens,
            latency_ms=latency_ms,
            request_id=data.get("id"),
            cost_usd=round(self.cost_usd(tokens), 10),
        )

    # ----------------------------------------------------------------------------- cache/telemetry
    @staticmethod
    def _cache_key(body: dict[str, Any]) -> str:
        # Option order is part of the question (order-bias checks rely on it), so no key sorting.
        blob = json.dumps(
            {"jev": 1, "m": body["model"], "s": body["state"], "q": body["questions"]}, default=str
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        from finpilot.db import session_scope
        from finpilot.models import LLMCache

        try:
            with session_scope() as s:
                row = s.get(LLMCache, key)
                return json.loads(row.response) if row else None
        except Exception:  # noqa: BLE001
            return None

    def _cache_put(self, key: str, data: dict[str, Any]) -> None:
        from finpilot.db import session_scope
        from finpilot.models import LLMCache

        try:
            with session_scope() as s:
                s.merge(LLMCache(key=key, model=f"jev:{self.model}", response=json.dumps(data, default=str)))
        except Exception:  # noqa: BLE001
            log.debug("jev cache write failed", exc_info=True)

    def _record(
        self,
        agent: str,
        purpose: str,
        ev: Evaluation | None,
        *,
        success: bool,
        error: str | None = None,
        request: dict[str, Any] | None = None,
        response: Any = None,
    ) -> None:
        entry = {
            "agent": agent,
            "provider": self.provider,
            "model": ev.model if ev else self.model,
            "prompt_tokens": ev.input_tokens if ev else 0,
            "completion_tokens": 0,
            "latency_ms": ev.latency_ms if ev else 0,
            "cached": ev.cached if ev else False,
            "success": success,
            "kind": "jev",
            "cost_usd": ev.cost_usd if ev else 0.0,
        }
        log_step_call({**entry, "purpose": purpose})
        record_model_call(
            entry, run_id=current_run_id(), error=error, purpose=purpose, request=request, response=response
        )


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return min(8.0, float(resp.headers.get("retry-after", "")))
    except ValueError:
        return None


_client: JevClient | None = None


def get_jev() -> JevClient:
    global _client
    if _client is None:
        _client = JevClient()
    return _client


def set_jev(client: JevClient) -> None:
    """Swap the global client (tests inject one backed by an httpx.MockTransport)."""
    global _client
    _client = client
