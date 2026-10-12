"""Request bodies for the REST API."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field


class TransactionCreate(BaseModel):
    date: dt.date
    description: str = Field(min_length=1, max_length=500)
    amount: float = Field(description="Negative for spending, positive for income")
    time: dt.time | None = None
    category: str | None = None


class TransactionUpdate(BaseModel):
    category: str | None = None
    notes: str | None = None
    apply_to_merchant: bool = True


class RecurringUpdate(BaseModel):
    usage: Literal["unknown", "used", "rarely_used"] | None = None
    status: Literal["active", "cancelled"] | None = None


class GoalCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    target_amount: float = Field(gt=0)
    saved_amount: float = Field(ge=0, default=0)
    deadline: dt.date
    priority: int = Field(ge=1, le=10, default=1)


class GoalUpdate(BaseModel):
    name: str | None = None
    target_amount: float | None = Field(default=None, gt=0)
    saved_amount: float | None = Field(default=None, ge=0)
    deadline: dt.date | None = None
    priority: int | None = Field(default=None, ge=1, le=10)
    status: Literal["active", "achieved", "archived"] | None = None


class AdjustmentIn(BaseModel):
    type: Literal["one_time", "recurring", "category_change", "cancel_recurring", "income_change"]
    amount: float = 0
    kind: Literal["expense", "income"] = "expense"
    date: dt.date | None = None
    end_date: dt.date | None = None
    cadence: Literal["weekly", "biweekly", "monthly", "quarterly", "yearly"] = "monthly"
    category: str | None = None
    pct: float | None = None
    merchants: list[str] = Field(default_factory=list)
    label: str = ""


class SimulationRequest(BaseModel):
    adjustments: list[AdjustmentIn] = Field(min_length=1, max_length=12)


class RecommendationUpdate(BaseModel):
    status: Literal["new", "accepted", "dismissed", "done"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=64)


class RunRequest(BaseModel):
    recategorize: bool = False
    deliver_digest: bool = False


class ProfileUpdate(BaseModel):
    name: str | None = None
    currency: str | None = Field(default=None, max_length=8)
    current_balance: float | None = None
    balance_as_of: dt.date | None = None
    safety_buffer: float | None = Field(default=None, ge=0)
    digest_email: str | None = None
    telegram_chat_id: str | None = None
    digest_enabled: bool | None = None


class AgentLLMOverride(BaseModel):
    provider: str | None = None
    model: str | None = None


class LLMSettingsUpdate(BaseModel):
    enabled: bool | None = None
    provider: str | None = None
    model: str | None = None
    agents: dict[str, AgentLLMOverride] | None = None


class LLMTestRequest(BaseModel):
    provider: str | None = None
    model: str | None = None


class SchedulerUpdate(BaseModel):
    digest_cron: str


class DemoReset(BaseModel):
    persona: Literal["young_professional", "family", "freelancer"] = "young_professional"
    seed: int | None = None
    run_pipeline: bool = True


def dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(exclude_unset=True)
