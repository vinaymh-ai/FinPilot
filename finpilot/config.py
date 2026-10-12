"""Application settings, loaded from environment variables / a local `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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

    @property
    def fallback_provider_list(self) -> list[str]:
        return [p.strip() for p in self.llm_fallback_providers.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
