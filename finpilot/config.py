"""Application settings, loaded from environment variables / a local `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- App -------------------------------------------------------------------------------
    app_name: str = "FinPilot"
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = False
    database_url: str = f"sqlite:///{ROOT_DIR / 'finpilot.db'}"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    frontend_dist: str = str(ROOT_DIR / "frontend" / "dist")
    seed_demo_data: bool = True
    api_token: str | None = Field(default=None, description="Optional bearer token protecting the API")

    # --- LLM routing -----------------------------------------------------------------------
    # Mesh API is the default (and the hackathon-mandated) gateway. Other providers are
    # supported for local development / portability; all speak the OpenAI wire protocol.
    llm_enabled: bool = True
    llm_provider: str = "mesh"
    llm_model: str | None = None
    llm_fallback_providers: str = ""  # e.g. "groq,ollama" — tried in order if the primary fails
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.2
    llm_cache_enabled: bool = True

    # Per-agent model overrides (use a cheap model for bulk work, a stronger one for advice)
    categorizer_model: str | None = None
    advisor_model: str | None = None
    chat_model: str | None = None
    digest_model: str | None = None

    # Provider credentials (only the ones you use need to be set)
    mesh_api_key: str | None = None
    mesh_base_url: str = "https://api.meshapi.ai/v1"
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    anthropic_api_key: str | None = None
    together_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    custom_llm_base_url: str | None = None
    custom_llm_api_key: str | None = None
    custom_llm_model: str | None = None
    freellmapi_base_url: str = "http://localhost:3001/v1"
    freellmapi_api_key: str | None = None

    # --- System One decisions (Jev) ---------------------------------------------------------
    # Closed-set judgments (which category, which intent, is this rewrite faithful?) go to
    # TypeSafe's Jev — a decision model that returns calibrated probabilities instead of text.
    # Default route is Mesh's POST /v1/evaluate, so every AI call still goes through Mesh.
    jev_enabled: bool = False
    jev_provider: str = "mesh"  # mesh (POST /v1/evaluate) | typesafe (direct TypeSafe API)
    jev_model: str = "typesafe/jev-1.13"  # pin the version: the "-latest" alias can change answers
    jev_timeout_s: float = 20.0
    jev_max_retries: int = 2
    jev_max_concurrency: int = 8
    jev_cache_enabled: bool = True
    mesh_api_version: str = "2026-08"
    typesafe_api_key: str | None = None
    typesafe_base_url: str = "https://api.typesafe.ai/v1"
    # Confidence policy. Defaults are conservative; tune them on held-out data
    # (`python -m evaluation.run_eval --jev` reports calibration and a suggested threshold).
    jev_accept_confidence: float = 0.5  # act on a Choice at/above this confidence
    jev_route_confidence: float = 0.6  # chat router answers without an LLM at/above this
    jev_guard_confidence: float = 0.6  # faithfulness guard needs this much certainty to accept LLM text
    jev_order_check: bool = True  # re-ask mid-confidence answers with options reversed (order bias)
    categorizer_escalate_to_llm: bool = False  # cascade low-confidence answers to the LLM (validate first)
    chat_fast_model: str | None = None  # LLM for simple escalated questions (Jev complexity score)
    # Dense embeddings for chat retrieval (optional). Any OpenAI-compatible /embeddings model, e.g.
    # "openai/text-embedding-3-small" on Mesh, "text-embedding-3-small" on OpenAI, "gemini-embedding-001" on
    # Gemini or FreeLLMAPI, "nomic-embed-text" on Ollama. EMBEDDING_PROVIDER picks the gateway (default:
    # LLM_PROVIDER). Pin one model, not "auto". Unset model → local retrieval only.
    embedding_model: str | None = None
    embedding_provider: str | None = None
    embedding_timeout_s: float = 8.0  # embeddings are optional: a slow endpoint must never stall chat
    # Prices used only for cost reporting (USD per 1M tokens)
    jev_price_in_per_1m: float = 0.042
    llm_price_in_per_1m: float = 0.15
    llm_price_out_per_1m: float = 0.60

    # --- Observability -----------------------------------------------------------------------
    log_file: str = str(ROOT_DIR / "logs" / "finpilot.log")  # JSON lines, rotated; "" disables the file
    log_file_max_mb: int = 20
    log_file_backups: int = 5
    trace_payloads: bool = True  # store raw model request/response (clipped) for debugging
    trace_max_chars: int = 20000  # per string field in a stored payload
    trace_retention_days: int = 14  # payloads older than this are purged nightly
    langsmith_tracing: bool = False  # opt-in: traces leave your machine (they contain financial data)
    langsmith_api_key: str | None = None
    langsmith_project: str = "finpilot"
    langsmith_endpoint: str | None = None

    # --- Accounts & authentication ------------------------------------------------------------
    auth_enabled: bool = True  # false = single-user mode (no login; everything belongs to user 1)
    allow_signup: bool = True
    session_ttl_days: int = 14
    cookie_secure: bool | None = None  # None = Secure cookies when the request came over HTTPS
    admin_emails: str = ""  # comma-separated; otherwise the first account created becomes admin
    demo_accounts: bool = True  # "Try the demo" creates a private, seeded account per visitor
    demo_ttl_hours: int = 48
    public_base_url: str | None = None  # used in password-reset emails (defaults to the request's origin)
    password_min_length: int = 8

    # --- Finance defaults -------------------------------------------------------------------
    currency: str = "INR"
    timezone: str = "Asia/Kolkata"
    forecast_horizon_days: int = 180
    monte_carlo_paths: int = 400
    safety_buffer: float | None = None  # minimum balance considered "safe"; auto-derived if unset

    # --- Scheduler ---------------------------------------------------------------------------
    scheduler_enabled: bool = True
    digest_cron: str = "0 9 * * mon"  # every Monday 09:00 (in `timezone`)
    pipeline_cron: str = "30 2 * * *"  # nightly re-analysis

    # --- Notifications -------------------------------------------------------------------------
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    digest_email_to: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @field_validator("cookie_secure", "safety_buffer", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        # `COOKIE_SECURE=` in a .env means "not set", not an invalid boolean
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def admin_email_list(self) -> list[str]:
        return [e.strip().lower() for e in self.admin_emails.split(",") if e.strip()]

    @property
    def fallback_provider_list(self) -> list[str]:
        return [p.strip() for p in self.llm_fallback_providers.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
