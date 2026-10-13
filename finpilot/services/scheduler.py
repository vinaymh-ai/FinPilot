"""Background scheduler (APScheduler) — the proactive part of FinPilot.

Two cron jobs, both configurable, plus housekeeping:
* `weekly_digest`   — re-runs the full agent graph and delivers the digest (in-app/email/Telegram)
* `nightly_refresh` — re-runs analysis + forecast so the dashboard and advice stay current
* `maintenance`     — purges old raw model payloads, expired sessions and expired demo accounts
Run exactly one scheduler per deployment (SCHEDULER_ENABLED=false on extra API replicas).
"""

from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from finpilot.config import get_settings
from finpilot.db import session_scope
from finpilot.models import User

log = logging.getLogger(__name__)


class SchedulerService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._sched: BackgroundScheduler | None = None
        self.digest_cron = self.settings.digest_cron
        self.pipeline_cron = self.settings.pipeline_cron

    @property
    def running(self) -> bool:
        return bool(self._sched and self._sched.running)

    def start(self, digest_cron: str | None = None) -> None:
        if self.running:
            return
        if digest_cron:
            self.digest_cron = digest_cron
        self._sched = BackgroundScheduler(
            timezone=self.settings.timezone,
            job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600},
        )
        self._sched.add_job(
            self.digest_job,
            CronTrigger.from_crontab(self.digest_cron, timezone=self.settings.timezone),
            id="weekly_digest",
            name="Weekly digest",
            replace_existing=True,
        )
        self._sched.add_job(
            self.pipeline_job,
            CronTrigger.from_crontab(self.pipeline_cron, timezone=self.settings.timezone),
            id="nightly_refresh",
            name="Nightly re-analysis",
            replace_existing=True,
        )
        self._sched.add_job(
            run_maintenance,
            CronTrigger(hour=3, minute=17, timezone=self.settings.timezone),
            id="maintenance",
            name="Housekeeping",
            replace_existing=True,
        )
        self._sched.start()
        log.info(
            "scheduler started (digest: %s, refresh: %s, tz=%s)",
            self.digest_cron,
            self.pipeline_cron,
            self.settings.timezone,
        )

    def shutdown(self) -> None:
        if self._sched and self._sched.running:
            self._sched.shutdown(wait=False)

    def reschedule_digest(self, cron: str) -> None:
        CronTrigger.from_crontab(cron, timezone=self.settings.timezone)  # validate first
        self.digest_cron = cron
        if self._sched:
            self._sched.reschedule_job(
                "weekly_digest", trigger=CronTrigger.from_crontab(cron, timezone=self.settings.timezone)
            )

    def jobs(self) -> list[dict[str, Any]]:
        if not self._sched:
            return []
        return [
            {
                "id": j.id,
                "name": j.name,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
                "trigger": str(j.trigger),
            }
            for j in self._sched.get_jobs()
        ]

    # ------------------------------------------------------------------ jobs
    @staticmethod
    def _user_ids(digest_only: bool) -> list[int]:
        with session_scope() as s:
            # demo sandboxes are short-lived and should not spend model budget on schedules
            q = select(User.id).where(User.is_demo.is_not(True), User.disabled.is_not(True))
            if digest_only:
                q = q.where(User.digest_enabled.is_(True))
            return list(s.scalars(q))

    def digest_job(self) -> None:
        from finpilot.agents.coordinator import Coordinator

        for uid in self._user_ids(digest_only=True):
            try:
                Coordinator().run_pipeline(uid, trigger="schedule", deliver_digest=True)
            except Exception:  # noqa: BLE001
                log.exception("scheduled digest failed for user %s", uid)

    def pipeline_job(self) -> None:
        from finpilot.agents.coordinator import Coordinator

        for uid in self._user_ids(digest_only=False):
            try:
                Coordinator().run_pipeline(uid, trigger="schedule")
            except Exception:  # noqa: BLE001
                log.exception("scheduled refresh failed for user %s", uid)


def run_maintenance() -> dict[str, Any]:
    """Retention: drop raw payloads past TRACE_RETENTION_DAYS, expired sessions and demo accounts."""
    import datetime as dt

    from sqlalchemy import update

    from finpilot.models import LLMCall
    from finpilot.services.auth import purge_expired

    s = get_settings()
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=s.trace_retention_days)
    result: dict[str, Any] = {}
    try:
        with session_scope() as db:
            res = db.execute(
                update(LLMCall)
                .where(LLMCall.created_at < cutoff, LLMCall.request_payload.is_not(None))
                .values(request_payload=None, response_payload=None)
            )
            result["payloads_purged"] = res.rowcount or 0
            result.update(purge_expired(db))
    except Exception:  # noqa: BLE001
        log.exception("maintenance failed")
        return result
    if any(result.values()):
        log.info("maintenance: %s", result, extra={"event": "maintenance", **result})
    return result


scheduler = SchedulerService()
