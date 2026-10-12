"""Profile, LLM provider configuration (bring-your-own provider), model discovery and demo data."""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from finpilot.agents.coordinator import Coordinator
from finpilot.api.deps import current_user_id, get_db, require_admin
from finpilot.api.routes.decisions import jev_state
from finpilot.config import get_settings
from finpilot.llm.client import AGENTS, LLMError, LLMRouter, get_router
from finpilot.llm.providers import PROVIDERS, is_configured, resolve_api_key, resolve_base_url
from finpilot.schemas import DemoReset, LLMSettingsUpdate, LLMTestRequest, ProfileUpdate, dump
from finpilot.services import repository as repo
from finpilot.services.seed import DEMO_SEED, seed_demo

router = APIRouter()


def _llm_state() -> dict[str, Any]:
    r = get_router()
    s = get_settings()
    providers = []
    for spec in PROVIDERS.values():
        providers.append(
            {
                "name": spec.name,
                "label": spec.label,
                "configured": is_configured(spec, s),
                "default_model": spec.default_model,
                "suggested_models": spec.suggested_models,
                "free_tier": spec.free_tier,
                "notes": spec.notes,
                "base_url": resolve_base_url(spec, s) or None,
                "key_env": f"{spec.key_attr.upper()}" if spec.key_attr else None,
            }
        )
    agents = {}
    for a in AGENTS:
        targets = r.targets(a) if r.enabled else []
        agents[a] = {
            "resolved": f"{targets[0][0].name}:{targets[0][1]}" if targets else "rules",
            "fallbacks": [f"{t[0].name}:{t[1]}" for t in targets[1:]],
        }
    return {
        "enabled": r.enabled,
        "env_provider": s.llm_provider,
        "overrides": r.overrides,
        "providers": providers,
        "agents": agents,
        "fallback_providers": s.fallback_provider_list,
    }


@router.get("/settings")
def get_settings_view(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    user = repo.get_or_create_user(db, uid)
    balance, as_of = repo.current_balance(db, user)
    return {
        "profile": {
            "name": user.name,
            "currency": user.currency,
            "current_balance": balance,
            "balance_as_of": as_of.isoformat(),
            "safety_buffer": user.safety_buffer,
            "digest_email": user.digest_email,
            "telegram_chat_id": user.telegram_chat_id,
            "digest_enabled": user.digest_enabled,
        },
        "llm": _llm_state(),
        "decisions": jev_state(),
        "notifications": {
            "email_configured": bool(get_settings().smtp_host),
            "telegram_configured": bool(get_settings().telegram_bot_token),
        },
    }


@router.put("/settings/profile")
def update_profile(
    body: ProfileUpdate, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    user = repo.get_or_create_user(db, uid)
    data = dump(body)
    if "current_balance" in data:
        # user states "my balance today is X": anchor it at the latest transaction date
        _, as_of = repo.current_balance(db, user)
        user.current_balance = data.pop("current_balance")
        user.balance_as_of = data.pop("balance_as_of", None) or as_of
    for k, v in data.items():
        setattr(user, k, v)
    db.commit()
    return get_settings_view(uid, db)


@router.put("/settings/llm", dependencies=[Depends(require_admin)])
def update_llm(body: LLMSettingsUpdate, db: Session = Depends(get_db)) -> dict[str, Any]:
    r = get_router()
    ov = r.overrides
    data = body.model_dump(exclude_unset=True)
    if "provider" in data and data["provider"] and data["provider"] not in PROVIDERS:
        raise HTTPException(422, f"unknown provider {data['provider']}")
    for k in ("enabled", "provider", "model"):
        if k in data:
            ov[k] = data[k]
    if "agents" in data and data["agents"] is not None:
        agents = dict(ov.get("agents") or {})
        for name, cfg in data["agents"].items():
            if name not in AGENTS:
                raise HTTPException(422, f"unknown agent {name}")
            clean = {k: v for k, v in (cfg or {}).items() if v}
            if clean:
                agents[name] = clean
            else:
                agents.pop(name, None)
        ov["agents"] = agents
    ov = {k: v for k, v in ov.items() if v not in (None, "", {})}
    r.set_overrides(ov)
    repo.put_setting(db, "llm_overrides", ov)
    db.commit()
    return _llm_state()


@router.post("/settings/llm/test", dependencies=[Depends(require_admin)])
def test_llm(body: LLMTestRequest) -> dict[str, Any]:
    base = get_router()
    provider = body.provider or base.overrides.get("provider") or get_settings().llm_provider
    if provider not in PROVIDERS:
        raise HTTPException(422, f"unknown provider {provider}")
    probe = LLMRouter(get_settings())
    probe.set_overrides({"enabled": True, "provider": provider, "model": body.model or None})
    if not probe.targets("chat"):
        return {"ok": False, "provider": provider, "error": "provider not configured (missing API key / URL)"}
    t0 = time.perf_counter()
    try:
        resp = probe.chat(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            agent="chat",
            max_tokens=5,
            temperature=0,
            cache=False,
        )
        return {
            "ok": True,
            "provider": provider,
            "model": resp.model,
            "reply": (resp.content or "").strip()[:40],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    except LLMError as exc:
        return {"ok": False, "provider": provider, "error": str(exc)[:400]}


@router.get("/llm/models")
def list_models(provider: str = "mesh") -> dict[str, Any]:
    """Live model catalogue from the provider's OpenAI-compatible /models endpoint."""
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise HTTPException(404, "unknown provider")
    s = get_settings()
    base = resolve_base_url(spec, s)
    key = resolve_api_key(spec, s)
    if not base:
        return {"provider": provider, "models": spec.suggested_models, "source": "suggested"}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    url = base.rstrip("/") + ("/models" if key or provider == "ollama" else "/public/models")
    try:
        r = httpx.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = data.get("data", data) if isinstance(data, dict) else data
        ids = sorted(
            {
                m.get("id")
                for m in rows
                if isinstance(m, dict) and m.get("id") and m.get("model_type", "text") in ("text", None)
            }
        )
        return {"provider": provider, "models": ids[:500], "source": "live"}
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "provider": provider,
            "models": spec.suggested_models,
            "source": "suggested",
            "error": str(exc)[:200],
        }


@router.post("/demo/reset")
async def reset_demo(
    body: DemoReset, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    info = seed_demo(db, uid, persona=body.persona, seed=body.seed if body.seed is not None else DEMO_SEED)
    result = None
    if body.run_pipeline:
        result = await run_in_threadpool(Coordinator().run_pipeline, uid, trigger="manual")
    return {"seeded": info, "run": result}
