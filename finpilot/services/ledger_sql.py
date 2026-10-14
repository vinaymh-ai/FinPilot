"""A sandboxed, per-user SQL view of the ledger for the chat's analyst agent.

Each request builds an **in-memory SQLite database that contains only the signed-in user's rows**
(transactions, recurring series, goals, latest insights and recommendations). Generated SQL can
therefore never reach another user's data or the application database, whatever it says. On top of
that isolation:

* an authorizer allows only SELECT / READ / FUNCTION / RECURSIVE actions (no PRAGMA, ATTACH, DML/DDL),
* ``PRAGMA query_only`` is set after loading, extension loading stays disabled,
* string/SQL length limits and a progress handler bound memory and CPU per query,
* results are capped (rows and cell size) before they reach a prompt or the browser.

Columns are designed so common money questions are one WHERE clause away: ``direction``
(debit/credit), ``abs_amount``, ``is_spending`` (consumption, excluding transfers and investments),
``transaction_kind``, ``channel``, ``month``/``weekday``/``hour`` and review/anomaly flags.
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from finpilot.domain.normalize import CHANNELS, detect_channel, direction_of, payment_provider
from finpilot.domain.recurring import monthly_equivalent
from finpilot.domain.stats import add_months
from finpilot.domain.taxonomy import BY_ID, CATEGORIES, is_spending_flow, label, transaction_kind

if TYPE_CHECKING:
    from finpilot.services.toolkit import FinanceToolkit

MAX_ROWS = 500
MAX_CELL_CHARS = 300
MAX_VM_STEPS = 4_000_000  # progress-handler budget (~0.5 s on a laptop)
_PROGRESS_EVERY = 1000

SCHEMA_DOC = """Tables (SQLite dialect). Every row belongs to the current user; no user filter is needed.

transactions — one row per bank transaction
  id INTEGER, date TEXT 'YYYY-MM-DD', time TEXT 'HH:MM' or NULL, hour INTEGER 0-23 or NULL,
  weekday TEXT ('Monday'..'Sunday'), weekday_num INTEGER (0=Monday..6=Sunday), is_weekend INTEGER 0/1,
  month TEXT 'YYYY-MM', year INTEGER,
  description TEXT (raw bank narration), merchant TEXT (cleaned payee name),
  amount REAL (SIGNED: negative = money out / debit, positive = money in / credit),
  abs_amount REAL (always positive), direction TEXT ('debit' | 'credit'),
  channel TEXT ('upi','card','netbanking','neft','imps','rtgs','atm','auto_debit','cheque','cash_deposit','bank','wallet','other'),
  category TEXT (category id, see categories), category_label TEXT, category_group TEXT
    ('income','essential','discretionary','financial','other'),
  transaction_kind TEXT ('expense','refund','transfers','credit_card_payment','investment','withdrawal','income','unknown'),
  is_spending INTEGER 0/1 (1 = consumption flow: expense debits and their refunds; transfers, investments and income are 0),
  category_source TEXT, category_confidence REAL, category_reason TEXT, needs_review INTEGER 0/1,
  payment_provider TEXT or NULL (CRED, Kiwi, Razorpay, ... — a payment channel, not a purpose),
  is_recurring INTEGER 0/1, is_anomaly INTEGER 0/1, anomaly_reason TEXT, is_impulse INTEGER 0/1,
  tags TEXT (comma-separated, e.g. 'impulse,late_night'), notes TEXT
categories — id TEXT, label TEXT, grp TEXT, description TEXT
recurring — merchant, category, kind ('subscription','bill','rent','emi','income','insurance','investment','other'),
  cadence ('weekly','monthly','quarterly','yearly'), amount REAL (positive), monthly_cost REAL, status ('active','lapsed','cancelled'),
  first_date, last_date, next_expected TEXT dates, occurrences INTEGER, price_change_pct REAL, usage_signal TEXT
goals — name, target REAL, saved REAL, remaining REAL, deadline TEXT, priority INTEGER, goal_type TEXT
insights — kind, severity ('critical','warning','info','positive'), title, detail (latest analysis run)
recommendations — title, message, action, monthly_impact REAL, trigger TEXT, status, urgency (latest advice)

Money semantics:
- "spent"/"spending" on something = SUM(abs_amount) over is_spending = 1 AND direction = 'debit'
  (report refunds separately: is_spending = 1 AND direction = 'credit').
- money received / income / credits = direction = 'credit' (income only: category = 'income').
- all money out of the bank (cash outflow) = direction = 'debit' (includes transfers and investments).
- Merchant names: match with merchant LIKE '%name%' (case-insensitive in SQLite for ASCII)."""


class LedgerSQLError(ValueError):
    """Generated SQL was rejected or failed; the message is safe to show to a model for repair."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": self.elapsed_ms,
        }


_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}
_FORBIDDEN_FUNCTIONS = {"load_extension", "randomblob", "zeroblob", "readfile", "writefile", "fts3_tokenizer"}
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _authorizer(
    action: int, arg1: str | None, arg2: str | None, _db: str | None, _trigger: str | None
) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() in _FORBIDDEN_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def clean_sql(sql: str) -> str:
    """Strip fences/comments, require exactly one SELECT/WITH statement."""
    text = (sql or "").strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"--[^\n]*", " ", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = text.strip().rstrip(";").strip()
    if not text:
        raise LedgerSQLError("empty query")
    if ";" in re.sub(r"'(?:[^']|'')*'", "''", text):
        raise LedgerSQLError("only one SQL statement is allowed")
    if not re.match(r"(?is)^\s*(select|with)\b", text):
        raise LedgerSQLError("only SELECT queries (optionally starting with WITH) are allowed")
    if re.search(
        r"(?i)\b(attach|detach|pragma|insert|update|delete|replace|create|drop|alter|vacuum|reindex)\b", text
    ):
        raise LedgerSQLError("the query must only read data")
    return text


class LedgerSQL:
    """One user's ledger as an isolated, read-only, in-memory SQLite database."""

    def __init__(self, kit: FinanceToolkit):
        self.kit = kit
        self.today = kit.today
        self.as_of = kit.as_of
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._load()
        for limit, value in (
            (sqlite3.SQLITE_LIMIT_LENGTH, 2_000_000),
            (sqlite3.SQLITE_LIMIT_SQL_LENGTH, 20_000),
            (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 200),
            (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 20),
        ):
            self.conn.setlimit(limit, value)
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.set_authorizer(_authorizer)

    # ------------------------------------------------------------------ build
    def _load(self) -> None:
        c = self.conn
        c.execute(
            """CREATE TABLE transactions (
                id INTEGER PRIMARY KEY, date TEXT, time TEXT, hour INTEGER, weekday TEXT, weekday_num INTEGER,
                is_weekend INTEGER, month TEXT, year INTEGER, description TEXT, merchant TEXT, amount REAL,
                abs_amount REAL, direction TEXT, channel TEXT, category TEXT, category_label TEXT,
                category_group TEXT, transaction_kind TEXT, is_spending INTEGER, category_source TEXT,
                category_confidence REAL, category_reason TEXT, needs_review INTEGER, payment_provider TEXT,
                is_recurring INTEGER, is_anomaly INTEGER, anomaly_reason TEXT, is_impulse INTEGER, tags TEXT,
                notes TEXT)"""
        )
        rows = []
        for t in self.kit.rows:
            amount = float(t.amount)
            cat = t.category or "other"
            tags = list(t.tags or [])
            rows.append(
                (
                    t.id,
                    t.date.isoformat(),
                    t.time.strftime("%H:%M") if t.time else None,
                    t.time.hour if t.time else None,
                    _WEEKDAYS[t.date.weekday()],
                    t.date.weekday(),
                    int(t.date.weekday() >= 5),
                    t.date.strftime("%Y-%m"),
                    t.date.year,
                    t.description,
                    t.merchant,
                    round(amount, 2),
                    round(abs(amount), 2),
                    t.direction or direction_of(amount),
                    t.channel or detect_channel(t.description),
                    cat,
                    label(cat),
                    BY_ID[cat].group if cat in BY_ID else "other",
                    transaction_kind(cat, amount),
                    int(is_spending_flow(cat, amount)),
                    t.category_source,
                    t.category_confidence,
                    t.category_reason,
                    int(cat == "other" or (t.category_confidence or 0) < 0.7),
                    payment_provider(t.description),
                    int(bool(t.is_recurring)),
                    int(bool(t.is_anomaly)),
                    t.anomaly_reason,
                    int("impulse" in tags),
                    ",".join(tags) or None,
                    t.notes,
                )
            )
        c.executemany(f"INSERT INTO transactions VALUES ({','.join('?' * 31)})", rows)
        c.execute("CREATE INDEX ix_t_date ON transactions(date)")
        c.execute("CREATE INDEX ix_t_merchant ON transactions(merchant)")
        c.execute("CREATE TABLE categories (id TEXT, label TEXT, grp TEXT, description TEXT)")
        c.executemany(
            "INSERT INTO categories VALUES (?,?,?,?)",
            [(x.id, x.label, x.group, x.description) for x in CATEGORIES],
        )
        self._load_context()
        c.commit()

    def _load_context(self) -> None:
        from finpilot.models import RecurringSeries
        from finpilot.services import repository as repo

        c, kit = self.conn, self.kit
        c.execute(
            """CREATE TABLE recurring (merchant TEXT, category TEXT, kind TEXT, cadence TEXT, amount REAL,
               monthly_cost REAL, status TEXT, first_date TEXT, last_date TEXT, next_expected TEXT,
               occurrences INTEGER, price_change_pct REAL, usage_signal TEXT)"""
        )
        linked = {t.recurring_id for t in kit.rows if t.recurring_id is not None}
        series = [
            r
            for r in kit.session.query(RecurringSeries).filter_by(user_id=kit.user_id).all()
            if r.id in linked
        ]
        c.executemany(
            "INSERT INTO recurring VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    r.merchant,
                    r.category,
                    r.kind,
                    r.cadence,
                    abs(float(r.typical_amount)),
                    round(monthly_equivalent(abs(float(r.typical_amount)), r.cadence), 2),
                    r.status,
                    r.first_date.isoformat(),
                    r.last_date.isoformat(),
                    r.next_expected.isoformat(),
                    r.occurrences,
                    r.price_change_pct,
                    r.usage_signal,
                )
                for r in series
            ],
        )
        c.execute(
            "CREATE TABLE goals (name TEXT, target REAL, saved REAL, remaining REAL, deadline TEXT, "
            "priority INTEGER, goal_type TEXT)"
        )
        c.executemany(
            "INSERT INTO goals VALUES (?,?,?,?,?,?,?)",
            [
                (
                    g.name,
                    g.target,
                    g.saved,
                    round(max(0.0, g.target - g.saved), 2),
                    g.deadline.isoformat(),
                    g.priority,
                    "spend" if g.spend_at_deadline else "reserve",
                )
                for g in kit.goals
            ],
        )
        c.execute("CREATE TABLE insights (kind TEXT, severity TEXT, title TEXT, detail TEXT)")
        c.executemany(
            "INSERT INTO insights VALUES (?,?,?,?)",
            [(i.kind, i.severity, i.title, i.detail) for i in repo.latest_insights(kit.session, kit.user_id)],
        )
        c.execute(
            "CREATE TABLE recommendations (title TEXT, message TEXT, action TEXT, monthly_impact REAL, "
            "trigger TEXT, status TEXT, urgency TEXT)"
        )
        c.executemany(
            "INSERT INTO recommendations VALUES (?,?,?,?,?,?,?)",
            [
                (r.title, r.message, r.action, float(r.monthly_impact or 0), r.trigger, r.status, r.urgency)
                for r in repo.latest_recommendations(kit.session, kit.user_id)
            ],
        )

    # ------------------------------------------------------------------ query
    def execute(self, sql: str, *, max_rows: int = MAX_ROWS) -> QueryResult:
        text = clean_sql(sql)
        steps = {"n": 0}

        def budget() -> int:
            steps["n"] += _PROGRESS_EVERY
            return 1 if steps["n"] > MAX_VM_STEPS else 0

        self.conn.set_progress_handler(budget, _PROGRESS_EVERY)
        started = time.perf_counter()
        try:
            cur = self.conn.execute(text)
            fetched = cur.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as exc:
            if steps["n"] > MAX_VM_STEPS:
                raise LedgerSQLError("query took too long; simplify it or add filters") from exc
            raise LedgerSQLError(str(exc)) from exc
        finally:
            self.conn.set_progress_handler(None, 0)
        columns = [d[0] for d in cur.description or []]
        rows = [[_cell(v) for v in row] for row in fetched[:max_rows]]
        return QueryResult(
            sql=text,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=len(fetched) > max_rows,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def scalar(self, sql: str) -> Any:
        res = self.execute(sql, max_rows=1)
        return res.rows[0][0] if res.rows and res.rows[0] else None

    # ------------------------------------------------------------------ prompt context
    def periods(self) -> dict[str, dict[str, str]]:
        """Precomputed calendar ranges so a model never has to do date arithmetic."""
        a = self.today
        month_start = a.replace(day=1)
        last_month_start = add_months(month_start, -1)
        monday = a - dt.timedelta(days=a.weekday())

        def rng(s: dt.date, e: dt.date) -> dict[str, str]:
            return {"start": s.isoformat(), "end": e.isoformat()}

        first = min((t.date for t in self.kit.rows), default=a)
        return {
            "today": rng(a, a),
            "yesterday": rng(a - dt.timedelta(days=1), a - dt.timedelta(days=1)),
            "this_week": rng(monday, a),
            "last_week": rng(monday - dt.timedelta(days=7), monday - dt.timedelta(days=1)),
            "this_month": rng(month_start, a),
            "last_month": rng(last_month_start, month_start - dt.timedelta(days=1)),
            "last_30_days": rng(a - dt.timedelta(days=29), a),
            "last_90_days": rng(a - dt.timedelta(days=89), a),
            "last_3_months": rng(add_months(month_start, -3), month_start - dt.timedelta(days=1)),
            "last_6_months": rng(add_months(month_start, -6), month_start - dt.timedelta(days=1)),
            "this_year": rng(a.replace(month=1, day=1), a),
            "last_year": rng(dt.date(a.year - 1, 1, 1), dt.date(a.year - 1, 12, 31)),
            "all_history": rng(first, self.as_of),
        }

    def profile(self) -> dict[str, Any]:
        """Small data profile for prompts: coverage, counts, channels and categories in use."""
        cur = self.conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date), SUM(direction='debit'), SUM(direction='credit') FROM transactions"
        )
        n, first, last, debits, credits = cur.fetchone()
        cats = self.conn.execute(
            "SELECT category, COUNT(*) FROM transactions GROUP BY category ORDER BY COUNT(*) DESC"
        ).fetchall()
        channels = self.conn.execute(
            "SELECT channel, COUNT(*) FROM transactions GROUP BY channel ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {
            "transactions": n,
            "first_date": first,
            "last_date": last,
            "debits": debits or 0,
            "credits": credits or 0,
            "categories_in_use": {k: v for k, v in cats},
            "channels_in_use": {CHANNELS.get(k, k): v for k, v in channels},
            "currency": self.kit.currency,
        }

    def close(self) -> None:
        self.conn.close()


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "…"
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value
