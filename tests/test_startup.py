"""First-boot data and repeat starts, on SQLite or FINPILOT_TEST_DATABASE_URL."""

from finpilot.config import get_settings
from finpilot.db import session_scope
from finpilot.main import bootstrap
from finpilot.models import AgentRun, Goal, Transaction, User
from finpilot.services import auth


def test_first_boot_seeds_once_and_first_signup_inherits_data(accounts, monkeypatch):
    monkeypatch.setattr(get_settings(), "seed_demo_data", True)
    bootstrap()
    with session_scope() as db:
        counts = (db.query(Transaction).count(), db.query(Goal).count(), db.query(AgentRun).count())
        assert counts[0] > 500 and counts[1] > 0 and counts[2] == 1
        assert db.get(User, 1).password_hash is None
        first = auth.signup(db, "owner@example.com", "correct horse battery")
        assert first.id == 1 and first.is_admin
        second = auth.signup(db, "second@example.com", "correct horse battery")
        assert second.id > 1 and not second.is_admin
        assert db.query(Transaction).filter_by(user_id=second.id).count() == 0
        db.query(Transaction).first().description = "Preserve my edit"
    bootstrap()
    with session_scope() as db:
        assert (db.query(Transaction).count(), db.query(Goal).count(), db.query(AgentRun).count()) == counts
        assert db.query(Transaction).filter_by(description="Preserve my edit").count() == 1


def test_startup_does_not_seed_an_existing_empty_account(accounts, monkeypatch):
    monkeypatch.setattr(get_settings(), "seed_demo_data", True)
    with session_scope() as db:
        auth.signup(db, "owner@example.com", "correct horse battery")
    bootstrap()
    with session_scope() as db:
        assert db.query(Transaction).count() == 0


def test_startup_can_disable_seeding(accounts):
    bootstrap()
    with session_scope() as db:
        assert db.query(User).count() == 0
        assert db.query(Transaction).count() == 0
