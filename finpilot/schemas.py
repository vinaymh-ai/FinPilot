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


GoalType = Literal[
    "trip", "purchase", "event", "emergency_fund", "investment", "debt_payoff", "home", "education", "other"
]


class GoalCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    target_amount: float = Field(gt=0)
    saved_amount: float = Field(ge=0, default=0)
    deadline: dt.date
    priority: int = Field(ge=1, le=10, default=1)
    # Optional: leave unset and System One classifies the goal from its name
    goal_type: GoalType | None = None
    spend_at_deadline: bool | None = None


class GoalUpdate(BaseModel):
    name: str | None = None
    target_amount: float | None = Field(default=None, gt=0)
    saved_amount: float | None = Field(default=None, ge=0)
    deadline: dt.date | None = None
    priority: int | None = Field(default=None, ge=1, le=10)
    status: Literal["active", "achieved", "archived"] | None = None
    goal_type: GoalType | None = None
    spend_at_deadline: bool | None = None


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
    # Omit (or send null) to start a new conversation; the response carries its id.
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ChatSessionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    pinned: bool | None = None


class CategoryCreate(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    parent: str = Field(default="other", max_length=48)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    icon: str | None = Field(default=None, max_length=32, pattern=r"^[a-z0-9-]+$")
    description: str | None = Field(default=None, max_length=255)
    # optional quick rule: narration/payee words that should land in this category
    keywords: list[str] = Field(default_factory=list, max_length=20)


class CategoryUpdate(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=60)
    parent: str | None = Field(default=None, max_length=48)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    icon: str | None = Field(default=None, max_length=32, pattern=r"^[a-z0-9-]+$")
    description: str | None = Field(default=None, max_length=255)


class RuleBody(BaseModel):
    category: str = Field(max_length=48)
    terms: list[str] = Field(min_length=1, max_length=20)
    field: Literal["any", "merchant", "description"] = "any"
    match: Literal["contains", "word", "equals", "starts_with"] = "contains"
    direction: Literal["debit", "credit"] | None = None
    min_amount: float | None = Field(default=None, ge=0)
    max_amount: float | None = Field(default=None, ge=0)
    channel: str | None = Field(default=None, max_length=16)
    priority: int = Field(default=100, ge=0, le=1000)
    enabled: bool = True
    apply_to_existing: bool = True
    include_manual: bool = False


class BulkCategorize(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=2000)
    category: str | None = Field(default=None, max_length=48)  # None + approve=True confirms current labels
    approve: bool = False
    remember: bool = False  # also learn merchant → category for future imports


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


class JevSettingsUpdate(BaseModel):
    enabled: bool | None = None
    accept_confidence: float | None = Field(default=None, ge=0, le=1)
    route_confidence: float | None = Field(default=None, ge=0, le=1)
    guard_confidence: float | None = Field(default=None, ge=0, le=1)
    escalate_to_llm: bool | None = None
    order_check: bool | None = None


class SchedulerUpdate(BaseModel):
    digest_cron: str


class DemoReset(BaseModel):
    persona: Literal["young_professional", "family", "freelancer"] = "young_professional"
    seed: int | None = None
    run_pipeline: bool = True


def dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(exclude_unset=True)


# ---------------------------------------------------------------------------- accounts


class SignupRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)
    name: str | None = Field(default=None, max_length=120)
    issue_token: bool = False  # also return the session token (for API clients; browsers use the cookie)


class LoginRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)
    remember: bool = True
    issue_token: bool = False


class ForgotPasswordRequest(BaseModel):
    email: str = Field(max_length=254)


class ResetPasswordRequest(BaseModel):
    token: str = Field(max_length=200)
    password: str = Field(max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(max_length=256)
    new_password: str = Field(max_length=256)


class DeleteAccountRequest(BaseModel):
    password: str | None = Field(default=None, max_length=256)


class ClientErrorReport(BaseModel):
    event_id: str | None = Field(default=None, max_length=64)
    message: str = Field(default="", max_length=2000)
    stack: str | None = Field(default=None, max_length=8000)
    url: str | None = Field(default=None, max_length=1000)
    user_agent: str | None = Field(default=None, max_length=400)
    context: dict[str, Any] | None = None
