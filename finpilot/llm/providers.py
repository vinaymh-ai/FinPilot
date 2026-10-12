"""Registry of OpenAI-compatible LLM providers.

Mesh API is the default gateway: one `rsk_` key reaches GPT, Claude, Gemini, Llama, … through
`https://api.meshapi.ai/v1`. The other entries exist so the same code runs against a local
Ollama model, a free-tier provider, or a self-hosted endpoint — every one of them speaks the
OpenAI chat-completions protocol, so switching is configuration, not code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finpilot.config import Settings

MESH_BASE_URL = "https://api.meshapi.ai/v1"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    base_url: str
    key_attr: str | None  # attribute on Settings holding the API key (None = no key needed)
    default_model: str
    suggested_models: list[str] = field(default_factory=list)
    free_tier: bool = False
    notes: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    supports_json_mode: bool = True


PROVIDERS: dict[str, ProviderSpec] = {
    "mesh": ProviderSpec(
        name="mesh",
        label="Mesh API (default gateway)",
        base_url=MESH_BASE_URL,
        key_attr="mesh_api_key",
        default_model="openai/gpt-4o-mini",
        suggested_models=[
            "openai/gpt-4o-mini",
            "google/gemini-3.5-flash",
            "anthropic/claude-haiku-4.5",
            "meta-llama/llama-3.3-70b-instruct",
            "openai/gpt-4o",
            "auto",
        ],
        notes="One rsk_ key, 500+ models, free 24h response cache. Hackathon-mandated gateway.",
    ),
    "openai": ProviderSpec(
        name="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        key_attr="openai_api_key",
        default_model="gpt-4o-mini",
        suggested_models=["gpt-4o-mini", "gpt-4o"],
    ),
    "groq": ProviderSpec(
        name="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_attr="groq_api_key",
        default_model="llama-3.3-70b-versatile",
        suggested_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        free_tier=True,
        notes="Generous free tier, very low latency.",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_attr="gemini_api_key",
        default_model="gemini-2.5-flash",
        suggested_models=["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        free_tier=True,
        notes="Free tier via Google AI Studio key.",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        key_attr="openrouter_api_key",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        suggested_models=["meta-llama/llama-3.3-70b-instruct:free", "deepseek/deepseek-chat-v3-0324:free"],
        free_tier=True,
        notes="':free' model variants cost nothing (rate-limited).",
        extra_headers={"HTTP-Referer": "https://github.com/vinaymh-ai/FinPilot", "X-Title": "FinPilot"},
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        label="Anthropic (OpenAI-compatible endpoint)",
        base_url="https://api.anthropic.com/v1/",
        key_attr="anthropic_api_key",
        default_model="claude-haiku-4-5",
        suggested_models=["claude-haiku-4-5", "claude-sonnet-4-5"],
        supports_json_mode=False,
    ),
    "together": ProviderSpec(
        name="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        key_attr="together_api_key",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        label="Ollama (local, free)",
        base_url="http://localhost:11434/v1",
        key_attr=None,
        default_model="llama3.1:8b",
        suggested_models=["llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
        free_tier=True,
        notes="Runs on your machine — zero cost, fully private.",
    ),
    "custom": ProviderSpec(
        name="custom",
        label="Custom OpenAI-compatible endpoint",
        base_url="",
        key_attr="custom_llm_api_key",
        default_model="default",
    ),
}


def resolve_base_url(spec: ProviderSpec, settings: Settings) -> str:
    if spec.name == "mesh":
        return settings.mesh_base_url or MESH_BASE_URL
    if spec.name == "ollama":
        return settings.ollama_base_url
    if spec.name == "custom":
        return settings.custom_llm_base_url or ""
    return spec.base_url


def resolve_api_key(spec: ProviderSpec, settings: Settings) -> str | None:
    if spec.key_attr is None:
        return "ollama"  # the OpenAI SDK requires a non-empty key; Ollama ignores it
    return getattr(settings, spec.key_attr, None)


def is_configured(spec: ProviderSpec, settings: Settings) -> bool:
    if spec.name == "ollama":
        return True  # assumed reachable if selected; failures fall through to deterministic mode
    if spec.name == "custom":
        return bool(settings.custom_llm_base_url)
    return bool(resolve_api_key(spec, settings))
