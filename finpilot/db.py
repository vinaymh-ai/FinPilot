"""Database engine / session management (SQLite by default, PostgreSQL via DATABASE_URL)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from finpilot.config import get_settings

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _make_engine(url: str) -> Engine:
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def configure_database(url: str | None = None) -> Engine:
    """(Re)configure the global engine. Tests call this with an in-memory / temp database."""
    global _engine, _SessionLocal
    _engine = _make_engine(url or get_settings().database_url)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        configure_database()
    assert _engine is not None
    return _engine


def init_db() -> None:
    from finpilot import models  # noqa: F401 - register tables

    engine = get_engine()
    Base.metadata.create_all(engine)
    added = _add_missing_columns(engine)
    _ensure_indexes(engine)
    _backfill(engine, added)
    _backfill_transaction_metadata(engine)
    _backfill_chat_sessions(engine)
    _load_custom_categories()


def _add_missing_columns(engine: Engine) -> list[str]:
    """Additive schema upgrade: create nullable columns that newer code expects.

    `create_all` creates missing *tables* but never alters existing ones, so a database created
    by an earlier version would lack new columns. Only nullable columns are added (no data
    rewrites); anything more invasive belongs in a real migration tool such as Alembic.
    """
    from sqlalchemy import inspect

    insp = inspect(engine)
    added: list[str] = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing or not col.nullable:
                    continue
                ddl_type = col.type.compile(dialect=engine.dialect)
                conn.exec_driver_sql(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl_type}')
                added.append(f"{table.name}.{col.name}")
    return added


def _ensure_indexes(engine: Engine) -> None:
    """Create indexes that newer code declares on tables an older version created."""
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            try:
                index.create(bind=engine, checkfirst=True)
            except Exception as exc:  # noqa: BLE001 - e.g. duplicate e-mails block a unique index
                log.warning("could not create index %s: %s", index.name, exc)


def _backfill(engine: Engine, added: list[str]) -> None:
    """Attribute model calls and decisions recorded before per-user tracking.

    Rows tied to an agent run take that run's user. Rows with no run (chat turns, imports) predate
    accounts, when FinPilot was single-user, so they belong to user 1 — applied only in the upgrade
    that adds the column, so later rows without a user are never silently re-attributed.
    """
    stmts: list[str] = []
    for table in ("llm_calls", "decisions"):
        stmts.append(
            f"UPDATE {table} SET user_id = (SELECT user_id FROM agent_runs WHERE agent_runs.id = {table}.run_id) "
            "WHERE user_id IS NULL AND run_id IS NOT NULL"
        )
        if f"{table}.user_id" in added:
            stmts.append(f"UPDATE {table} SET user_id = 1 WHERE user_id IS NULL")
    try:
        with engine.begin() as conn:
            for sql in stmts:
                conn.exec_driver_sql(sql)
    except Exception:  # noqa: BLE001 - best effort; never block startup
        log.warning("backfill of user ids failed", exc_info=True)


def _backfill_transaction_metadata(engine: Engine) -> None:
    """Derive debit/credit direction and payment channel for rows imported by older versions."""
    from sqlalchemy import or_, select, update

    from finpilot.domain.normalize import detect_channel, direction_of
    from finpilot.models import Transaction

    try:
        with engine.begin() as conn:
            rows = conn.execute(
                select(Transaction.id, Transaction.amount, Transaction.description).where(
                    or_(Transaction.direction.is_(None), Transaction.channel.is_(None))
                )
            ).all()
            for tid, amount, description in rows:
                conn.execute(
                    update(Transaction)
                    .where(Transaction.id == tid)
                    .values(direction=direction_of(float(amount)), channel=detect_channel(description or ""))
                )
        if rows:
            log.info("derived direction and channel for %s existing transactions", len(rows))
    except Exception:  # noqa: BLE001 - best effort; never block startup
        log.warning("transaction metadata backfill failed", exc_info=True)


def _backfill_chat_sessions(engine: Engine) -> None:
    """Give conversations recorded before chat threads existed a session row and a title."""
    from sqlalchemy import func, select

    from finpilot.models import ChatMessage, ChatSession

    try:
        with Session(engine) as s:
            known = set(s.execute(select(ChatSession.user_id, ChatSession.session_id)).tuples())
            rows = s.execute(
                select(
                    ChatMessage.session_id,
                    ChatMessage.user_id,
                    func.count(),
                    func.min(ChatMessage.created_at),
                    func.max(ChatMessage.created_at),
                ).group_by(ChatMessage.session_id, ChatMessage.user_id)
            ).all()
            for sid, uid, n, first, last in rows:
                if (uid, sid) in known:
                    continue
                q = s.scalar(
                    select(ChatMessage.content)
                    .where(
                        ChatMessage.session_id == sid, ChatMessage.user_id == uid, ChatMessage.role == "user"
                    )
                    .order_by(ChatMessage.id)
                    .limit(1)
                )
                from finpilot.agents.memory import title_from_question

                s.add(
                    ChatSession(
                        session_id=sid,
                        user_id=uid,
                        title=title_from_question(q or "Conversation"),
                        title_source="auto",
                        message_count=n,
                        created_at=first,
                        updated_at=last,
                    )
                )
                known.add((uid, sid))
            s.commit()
    except Exception:  # noqa: BLE001 - best effort; never block startup
        log.warning("chat session backfill failed", exc_info=True)


def _load_custom_categories() -> None:
    from finpilot.domain import taxonomy

    try:
        with new_session() as s:
            taxonomy.load_custom(s)
    except Exception:  # noqa: BLE001 - best effort; never block startup
        log.warning("could not load custom categories", exc_info=True)


def new_session() -> Session:
    if _SessionLocal is None:
        configure_database()
    assert _SessionLocal is not None
    return _SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = new_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = new_session()
    try:
        yield session
    finally:
        session.close()
