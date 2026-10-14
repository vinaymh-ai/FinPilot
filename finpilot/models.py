"""ORM models.

Money is stored as NUMERIC(14,2) and surfaced as float for analytics. Debits are negative,
credits positive. Every analytical artefact (insight, forecast, recommendation, digest) points
back to the `AgentRun` that produced it, which is what makes the decision trail auditable.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Time,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finpilot.db import Base

Money = Numeric(14, 2, asdecimal=False)


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (Index("uq_users_email", "email", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), default="Demo User")
    email: Mapped[str | None] = mapped_column(String(255))  # stored lower-cased; login identifier
    currency: Mapped[str] = mapped_column(String(8), default="INR")
    # Anchor balance: the account balance as of `balance_as_of`. Forecasts start from here.
    current_balance: Mapped[float] = mapped_column(Money, default=0.0)
    balance_as_of: Mapped[dt.date | None] = mapped_column(Date)
    balance_source: Mapped[str | None] = mapped_column(String(16))  # statement | user | demo | derived
    safety_buffer: Mapped[float | None] = mapped_column(Money)
    telegram_chat_id: Mapped[str | None] = mapped_column(String(64))
    digest_email: Mapped[str | None] = mapped_column(String(255))
    digest_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # --- account ------------------------------------------------------------------------------
    password_hash: Mapped[str | None] = mapped_column(String(255))  # scrypt; never returned by the API
    email_verified: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_admin: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_demo: Mapped[bool | None] = mapped_column(Boolean, default=False)
    disabled: Mapped[bool | None] = mapped_column(Boolean, default=False)
    last_login_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class AuthSession(Base):
    """A signed-in browser/API client. Only the SHA-256 of the token is stored."""

    __tablename__ = "auth_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    ip: Mapped[str | None] = mapped_column(String(64))


class AuthToken(Base):
    """One-time tokens (password reset). Hashed, expiring, single use."""

    __tablename__ = "auth_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    purpose: Mapped[str] = mapped_column(String(24))  # password_reset
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class ImportBatch(TimestampMixin, Base):
    __tablename__ = "import_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(32))  # csv | manual | demo
    filename: Mapped[str | None] = mapped_column(String(255))
    rows_total: Mapped[int] = mapped_column(Integer, default=0)
    rows_imported: Mapped[int] = mapped_column(Integer, default=0)
    rows_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    rows_invalid: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[list[str] | None] = mapped_column(JSON)


class Transaction(TimestampMixin, Base):
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint("user_id", "fingerprint", name="uq_txn_fingerprint"),
        Index("ix_txn_user_date", "user_id", "date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("import_batches.id", ondelete="SET NULL"))
    date: Mapped[dt.date] = mapped_column(Date)
    time: Mapped[dt.time | None] = mapped_column(Time)
    description: Mapped[str] = mapped_column(String(512))
    merchant: Mapped[str] = mapped_column(String(255), index=True)
    amount: Mapped[float] = mapped_column(Money)
    account: Mapped[str | None] = mapped_column(String(64))
    fingerprint: Mapped[str] = mapped_column(String(64))
    # Debit (money out) or credit (money in), and the payment rail read from the narration
    # (upi | card | netbanking | neft | imps | rtgs | atm | auto_debit | cheque | ...). Both are
    # filled automatically on insert/update, so every import path and agent sees them.
    direction: Mapped[str | None] = mapped_column(String(8), index=True)
    channel: Mapped[str | None] = mapped_column(String(16))

    category: Mapped[str | None] = mapped_column(String(48), index=True)
    category_source: Mapped[str | None] = mapped_column(
        String(24)
    )  # memory|rule|keyword|pattern|llm|user|fallback
    category_confidence: Mapped[float | None] = mapped_column(Float)
    category_reason: Mapped[str | None] = mapped_column(String(512))
    tags: Mapped[list[str] | None] = mapped_column(JSON)  # e.g. ["impulse", "late_night"]
    is_recurring: Mapped[bool] = mapped_column(Boolean, default=False)
    recurring_id: Mapped[int | None] = mapped_column(ForeignKey("recurring_series.id", ondelete="SET NULL"))
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    anomaly_reason: Mapped[str | None] = mapped_column(String(512))
    notes: Mapped[str | None] = mapped_column(Text)


class MerchantMemory(TimestampMixin, Base):
    """User corrections. The categorizer consults this first, so the system learns from edits."""

    __tablename__ = "merchant_memory"
    __table_args__ = (UniqueConstraint("user_id", "merchant_key", name="uq_merchant_memory"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    merchant_key: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(48))
    hits: Mapped[int] = mapped_column(Integer, default=0)


class RecurringSeries(TimestampMixin, Base):
    __tablename__ = "recurring_series"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    merchant: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(48))
    kind: Mapped[str] = mapped_column(
        String(24)
    )  # subscription|bill|rent|emi|income|insurance|investment|other
    cadence: Mapped[str] = mapped_column(String(16))  # weekly|monthly|quarterly|yearly
    interval_days: Mapped[float] = mapped_column(Float)
    typical_amount: Mapped[float] = mapped_column(Money)
    last_amount: Mapped[float] = mapped_column(Money)
    first_date: Mapped[dt.date] = mapped_column(Date)
    last_date: Mapped[dt.date] = mapped_column(Date)
    next_expected: Mapped[dt.date] = mapped_column(Date)
    occurrences: Mapped[int] = mapped_column(Integer)
    regularity: Mapped[float] = mapped_column(Float)  # 0..1, 1 = perfectly regular
    price_change_pct: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|lapsed|cancelled
    cancelled_on: Mapped[dt.date | None] = mapped_column(Date)
    usage: Mapped[str] = mapped_column(String(16), default="unknown")  # unknown|used|rarely_used (user-set)
    usage_signal: Mapped[str | None] = mapped_column(String(255))  # data-derived evidence of (non-)usage


class Goal(TimestampMixin, Base):
    __tablename__ = "goals"
    is_demo: Mapped[bool | None] = mapped_column(Boolean, default=False)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    target_amount: Mapped[float] = mapped_column(Money)
    saved_amount: Mapped[float] = mapped_column(Money, default=0.0)
    deadline: Mapped[dt.date] = mapped_column(Date)
    priority: Mapped[int] = mapped_column(Integer, default=1)  # 1 = highest
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|achieved|archived
    history: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)  # progress snapshots per run
    # What kind of goal it is decides how the forecast treats it: a "spend" goal (trip, laptop)
    # leaves the account at its deadline; a "reserve" goal (emergency fund) stays as savings.
    goal_type: Mapped[str | None] = mapped_column(String(32))
    spend_at_deadline: Mapped[bool | None] = mapped_column(Boolean)
    type_source: Mapped[str | None] = mapped_column(String(16))  # jev | rules | user
    type_confidence: Mapped[float | None] = mapped_column(Float)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    trigger: Mapped[str] = mapped_column(String(24))  # manual|upload|schedule|chat|digest|api
    graph: Mapped[str] = mapped_column(String(48), default="pipeline")
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|succeeded|failed
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    analysis: Mapped[dict[str, Any] | None] = mapped_column(
        JSON
    )  # analyzer snapshot (trends, subs, behaviour)
    input_signature: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)

    steps: Mapped[list[AgentStep]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="AgentStep.id"
    )


class AgentStep(Base):
    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    agent: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # succeeded|failed|skipped
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    run: Mapped[AgentRun] = relationship(back_populates="steps")


class LLMCall(TimestampMixin, Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"), index=True)
    agent: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str | None] = mapped_column(String(16), default="llm")  # llm (generative) | jev (decision)
    purpose: Mapped[str | None] = mapped_column(String(64))  # Jev: merchant_category, chat_route, …
    user_id: Mapped[int | None] = mapped_column(Integer, index=True)  # who the call was made for
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)  # HTTP request that caused it
    request_payload: Mapped[Any | None] = mapped_column(
        JSON(none_as_null=True)
    )  # raw request (clipped), see TRACE_PAYLOADS
    response_payload: Mapped[Any | None] = mapped_column(JSON(none_as_null=True))  # raw response (clipped)
    cost_usd: Mapped[float | None] = mapped_column(Float)  # estimated from configured prices


class Decision(TimestampMixin, Base):
    """One System One (Jev) judgment and what the code did with it — the decision audit log."""

    __tablename__ = "decisions"
    __table_args__ = (Index("ix_decision_purpose", "purpose", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"), index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, index=True)
    agent: Mapped[str] = mapped_column(String(32))
    purpose: Mapped[str] = mapped_column(String(48))  # merchant_category | chat_route | faithfulness | ...
    subject: Mapped[str] = mapped_column(String(255))  # what was decided about (merchant, message, …)
    question: Mapped[str] = mapped_column(String(64))
    answer: Mapped[str] = mapped_column(String(128))
    confidence: Mapped[float | None] = mapped_column(Float)
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSON)
    action: Mapped[str] = mapped_column(String(24))  # accepted | deferred | escalated | rejected | overridden
    model: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)


class MerchantProfile(TimestampMixin, Base):
    """What a model decided about a merchant, reused on every later import (pay once per merchant)."""

    __tablename__ = "merchant_profiles"
    __table_args__ = (UniqueConstraint("user_id", "merchant_key", name="uq_merchant_profile"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    merchant_key: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(48))
    confidence: Mapped[float] = mapped_column(Float)
    payee_type: Mapped[str | None] = mapped_column(String(32))  # business | person | institution
    service_family: Mapped[str | None] = mapped_column(String(48))  # video_streaming, music_streaming, …
    source: Mapped[str] = mapped_column(String(16))  # jev | llm
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSON)


class LLMCache(Base):
    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(128))
    response: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Insight(TimestampMixin, Base):
    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(48))
    severity: Mapped[str] = mapped_column(String(16))  # info|warning|critical|positive
    title: Mapped[str] = mapped_column(String(255))
    detail: Mapped[str] = mapped_column(Text)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    transaction_ids: Mapped[list[int] | None] = mapped_column(JSON)


class Forecast(TimestampMixin, Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    horizon_days: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class Recommendation(TimestampMixin, Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(48))
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(255))
    monthly_impact: Mapped[float] = mapped_column(Money, default=0.0)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    urgency: Mapped[str] = mapped_column(String(16), default="normal")  # low|normal|high
    status: Mapped[str] = mapped_column(String(16), default="new")  # new|accepted|dismissed|done
    # --- Decision trail -------------------------------------------------------------------
    trigger: Mapped[str] = mapped_column(String(64))  # which signal fired (pattern / forecast / goal)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON)  # the numbers + transaction ids behind it
    reasoning: Mapped[list[str]] = mapped_column(JSON)  # ordered reasoning steps
    generated_by: Mapped[str] = mapped_column(String(128))  # "rules" or "<provider>:<model>"
    grounded: Mapped[bool] = mapped_column(Boolean, default=True)  # passed the numeric grounding check
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)  # stable id across runs
    updated_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())


class Digest(TimestampMixin, Base):
    __tablename__ = "digests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"))
    trigger: Mapped[str] = mapped_column(String(24))  # schedule|manual
    subject: Mapped[str] = mapped_column(String(255))
    headline: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    body_markdown: Mapped[str] = mapped_column(Text)
    deliveries: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # channel -> {status, detail}
    recommendation_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendations.id", ondelete="SET NULL")
    )


class ChatMessage(TimestampMixin, Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user|assistant
    content: Mapped[str] = mapped_column(Text)
    trace: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)  # tool calls with args/results
    route: Mapped[dict[str, Any] | None] = mapped_column(
        JSON
    )  # System One routing decision (assistant turns)
    artifacts: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)  # exact tables / charts
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # model, agent pipeline, analyst/planner info


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)


@event.listens_for(Transaction, "before_insert")
@event.listens_for(Transaction, "before_update")
def _derive_transaction_metadata(_mapper: Any, _connection: Any, target: Transaction) -> None:
    """Keep direction (debit/credit) and payment channel consistent with amount and narration."""
    from finpilot.domain.normalize import detect_channel, direction_of

    if target.amount is not None:
        target.direction = direction_of(float(target.amount))
    if target.description is not None and (target.channel is None or _changed(target, "description")):
        target.channel = detect_channel(target.description)


def _changed(target: Any, attr: str) -> bool:
    from sqlalchemy import inspect as sa_inspect

    state = sa_inspect(target)
    return bool(state.attrs[attr].history.has_changes()) if state.persistent else False
