"""Demo account seeding."""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from finpilot.models import Goal
from finpilot.services import repository as repo
from finpilot.services.ingestion import ParsedRow, import_rows
from finpilot.services.synthetic import generate

log = logging.getLogger(__name__)

DEMO_PERSONA = "young_professional"
DEMO_SEED = 23


def seed_demo(
    session: Session,
    user_id: int = repo.DEFAULT_USER_ID,
    persona: str = DEMO_PERSONA,
    seed: int = DEMO_SEED,
    months: int = 8,
    end: dt.date | None = None,
) -> dict:
    ds = generate(persona, months=months, end=end or dt.date.today(), seed=seed)
    repo.wipe_user_data(session, user_id)
    user = repo.get_or_create_user(session, user_id)
    user.name = ds.display_name
    user.current_balance = ds.opening_balance
    user.balance_as_of = min(t.date for t in ds.transactions) - dt.timedelta(days=1)
    rows = [ParsedRow(t.date, t.time, t.description, t.amount) for t in ds.transactions]
    batch = import_rows(session, user, rows, source="demo", filename=f"demo-{persona}.csv")
    for g in ds.goals:
        session.add(
            Goal(
                user_id=user_id,
                name=g["name"],
                target_amount=g["target_amount"],
                saved_amount=g["saved_amount"],
                deadline=dt.date.fromisoformat(g["deadline"]),
                priority=g["priority"],
            )
        )
    session.commit()
    log.info("seeded demo persona %s: %s transactions", persona, batch.rows_imported)
    return {
        "persona": persona,
        "transactions": batch.rows_imported,
        "goals": len(ds.goals),
        "closing_balance": ds.closing_balance,
    }
