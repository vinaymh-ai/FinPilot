"""Coordinator — a LangGraph state machine that routes work to the specialist agents.

    START → plan ─┬─(new/uncategorised txns)→ categorize ─┐
                  ├─(nothing to do)────────────→ END       │
                  └────────────────────────────→ analyze ←─┘
                                                   │
                                     forecast ← ───┘
                                        │
                                      advise ─┬─(scheduled digest)→ digest → END
                                              └────────────────────────────→ END

Every node is wrapped so that it records an `AgentStep` (timing, summary, details, LLM calls,
errors) under one `AgentRun` — the run is the root of the decision trail. A failing non-critical
node degrades gracefully (e.g. no forecast → advice from patterns only); a failing critical node
(analysis) aborts the run with a recorded error.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from finpilot.agents.advisor import AdvisorAgent
from finpilot.agents.analyzer import AnalysisReport, AnalyzerAgent
from finpilot.agents.categorizer import CategorizerAgent, CatItem
from finpilot.agents.digest import DigestAgent
from finpilot.agents.forecaster import ForecasterAgent
from finpilot.config import get_settings
from finpilot.db import new_session
from finpilot.decisions.jev import JevClient, get_jev
from finpilot.domain.recurring import detect_recurring
from finpilot.llm.client import LLMRouter, get_router, run_context, step_call_log
from finpilot.models import AgentRun, AgentStep, Digest, Forecast, Insight, Recommendation, Transaction
from finpilot.services import notifier
from finpilot.services import repository as repo
from finpilot.services.goal_types import ensure_goal_types

log = logging.getLogger(__name__)

_user_locks: dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(user_id: int) -> threading.Lock:
    with _locks_guard:
        return _user_locks.setdefault(user_id, threading.Lock())


class PipelineState(TypedDict, total=False):
    user_id: int
    run_id: int
    trigger: str
    recategorize: bool
    deliver_digest: bool
    plan: list[str]
    n_transactions: int
    n_uncategorized: int
    categorization: dict[str, Any]
    analysis: AnalysisReport
    forecast: dict[str, Any]
    advice: dict[str, Any]
    digest_id: int
    errors: list[str]
    abort: bool


CRITICAL = {"analyze"}


class Coordinator:
    """Builds and runs the analyze → forecast → advise graph for one user."""

    AGENTS = {
        "categorizer": CategorizerAgent,
        "analyzer": AnalyzerAgent,
        "forecaster": ForecasterAgent,
        "advisor": AdvisorAgent,
        "digest": DigestAgent,
    }

    def __init__(
        self,
        router: LLMRouter | None = None,
        use_llm: bool = True,
        jev: JevClient | None = None,
        use_jev: bool = True,
    ):
        self.router = router or get_router()
        self.use_llm = use_llm
        self.jev = jev or get_jev()
        self.use_jev = use_jev
        self.settings = get_settings()

    # ------------------------------------------------------------------ public
    def run_pipeline(
        self,
        user_id: int = repo.DEFAULT_USER_ID,
        *,
        trigger: str = "manual",
        deliver_digest: bool = False,
        recategorize: bool = False,
    ) -> dict[str, Any]:
        lock = _lock_for(user_id)
        if not lock.acquire(timeout=120):
            raise RuntimeError("another pipeline run is in progress for this user")
        session = new_session()
        try:
            run = AgentRun(user_id=user_id, trigger=trigger, graph="pipeline", status="running")
            session.add(run)
            session.commit()
            graph = self.build_graph(session)
            state: PipelineState = {
                "user_id": user_id,
                "run_id": run.id,
                "trigger": trigger,
                "deliver_digest": deliver_digest,
                "recategorize": recategorize,
                "errors": [],
            }
            with run_context(run.id):
                final = graph.invoke(state)
            run = session.get(AgentRun, run.id)
            assert run is not None
            run.finished_at = dt.datetime.now(dt.UTC)
            run.status = "failed" if final.get("abort") else "succeeded"
            run.error = "; ".join(final.get("errors") or []) or None
            run.summary = self._summary(final)
            session.commit()
            return {
                "run_id": run.id,
                "status": run.status,
                "summary": run.summary,
                "errors": final.get("errors"),
            }
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            lock.release()

    def build_graph(self, session: Session):
        g = StateGraph(PipelineState)
        g.add_node("plan", self._wrap(session, "coordinator", "plan", self._plan))
        g.add_node("categorize", self._wrap(session, "categorizer", "categorize", self._categorize))
        g.add_node("analyze", self._wrap(session, "analyzer", "analyze", self._analyze))
        g.add_node("forecast", self._wrap(session, "forecaster", "forecast", self._forecast))
        g.add_node("advise", self._wrap(session, "advisor", "advise", self._advise))
        g.add_node("digest", self._wrap(session, "digest", "digest", self._digest))
        g.add_edge(START, "plan")
        g.add_conditional_edges(
            "plan", self._route_after_plan, {"categorize": "categorize", "analyze": "analyze", "end": END}
        )
        g.add_edge("categorize", "analyze")
        g.add_conditional_edges(
            "analyze", lambda s: "end" if s.get("abort") else "forecast", {"forecast": "forecast", "end": END}
        )
        g.add_edge("forecast", "advise")
        g.add_conditional_edges(
            "advise",
            lambda s: "digest" if s.get("deliver_digest") else "end",
            {"digest": "digest", "end": END},
        )
        g.add_edge("digest", END)
        return g.compile()

    @staticmethod
    def mermaid() -> str:
        return Coordinator(use_llm=False).build_graph(None).get_graph().draw_mermaid()  # type: ignore[arg-type]

    # ------------------------------------------------------------------ step wrapper
    def _wrap(
        self,
        session: Session,
        agent: str,
        node: str,
        fn: Callable[[Session, PipelineState], tuple[dict[str, Any], str, dict[str, Any]]],
    ):
        def runner(state: PipelineState) -> dict[str, Any]:
            started = dt.datetime.now(dt.UTC)
            t0 = time.perf_counter()
            with step_call_log() as calls:
                try:
                    update_, summary, details = fn(session, state)
                    status, error = "succeeded", None
                except Exception as exc:  # noqa: BLE001
                    session.rollback()
                    log.exception("node %s failed", node)
                    update_ = {"errors": (state.get("errors") or []) + [f"{node}: {exc}"]}
                    if node in CRITICAL:
                        update_["abort"] = True
                    summary, details = f"{node} failed: {exc}", {"traceback": traceback.format_exc()[-2000:]}
                    status, error = "failed", str(exc)
            details = dict(details or {})
            llm_calls = [c for c in calls if c.get("kind") != "jev"]
            jev_calls = [c for c in calls if c.get("kind") == "jev"]
            if llm_calls:
                details["llm_calls"] = llm_calls
            if jev_calls:
                details["jev_calls"] = _summarise_jev(jev_calls)
            session.add(
                AgentStep(
                    run_id=state["run_id"],
                    agent=agent,
                    status=status,
                    started_at=started,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    summary=summary,
                    details=_jsonable(details),
                    llm_calls=len(llm_calls),
                    error=error,
                )
            )
            session.commit()
            return update_

        return runner

    def _agent(self, name: str, **kwargs: Any):
        return self.AGENTS[name](
            router=self.router, use_llm=self.use_llm, jev=self.jev, use_jev=self.use_jev, **kwargs
        )

    # ------------------------------------------------------------------ nodes
    def _plan(self, session: Session, state: PipelineState):
        uid = state["user_id"]
        rows = repo.load_transactions(session, uid)
        n = len(rows)
        unc = sum(1 for t in rows if t.category is None)
        if n == 0:
            return {"plan": [], "n_transactions": 0}, "No transactions yet — nothing to analyse", {}
        plan = (["categorize"] if (unc or state.get("recategorize")) else []) + [
            "analyze",
            "forecast",
            "advise",
        ]
        if state.get("deliver_digest"):
            plan.append("digest")
        llm = (
            self.router.describe("advisor")
            if self.use_llm and self.router.available("advisor")
            else "rules only"
        )
        jev = self.jev.describe() if self.use_jev else "off"
        return (
            {"plan": plan, "n_transactions": n, "n_uncategorized": unc},
            f"{n} transactions ({unc} uncategorised) → plan: {' → '.join(plan)} "
            f"[decisions: {jev}; LLM: {llm}]",
            {"plan": plan, "transactions": n, "uncategorized": unc, "llm": llm, "decisions": jev},
        )

    @staticmethod
    def _route_after_plan(state: PipelineState) -> str:
        plan = state.get("plan") or []
        if not plan:
            return "end"
        return "categorize" if plan[0] == "categorize" else "analyze"

    def _categorize(self, session: Session, state: PipelineState):
        uid = state["user_id"]
        rows = repo.load_transactions(session, uid)
        if not state.get("recategorize"):
            targets = [t for t in rows if t.category is None]
        else:
            targets = [t for t in rows if t.category_source not in ("user",)]
        user = repo.get_or_create_user(session, uid)
        agent = self._agent(
            "categorizer",
            memory=repo.merchant_memory(session, uid),
            history=repo.merchant_history(session, uid),
            currency=user.currency,
        )
        items = [CatItem(t.id, t.description, float(t.amount), t.date) for t in targets]
        results, stats = agent.categorize(items)
        by_id = {t.id: t for t in targets}
        for r in results:
            t = by_id[r.key]
            t.merchant, t.category, t.category_source = r.merchant, r.category, r.source
            t.category_confidence, t.category_reason = r.confidence, r.reason
        repo.save_merchant_profiles(session, uid, agent.profiles)
        session.commit()
        st = stats.as_dict()
        src = ", ".join(f"{k} {v}" for k, v in sorted(st["by_source"].items(), key=lambda kv: -kv[1]))
        j = st["jev"]
        tail = ""
        if j["merchants"]:
            tail = (
                f"; Jev decided {j['merchants']} unknown merchants ({j['accepted']} accepted, "
                f"{j['deferred']} deferred to rules, ${j['cost_usd']:.5f})"
            )
        elif st["llm_merchants"]:
            tail = f"; LLM looked up {st['llm_merchants']} unknown merchants"
        return ({"categorization": st}, f"Categorised {len(results)} transactions ({src}){tail}", st)

    def _analyze(self, session: Session, state: PipelineState):
        uid = state["user_id"]
        user = repo.get_or_create_user(session, uid)
        rows = repo.load_transactions(session, uid)
        lite = [repo.to_lite(t) for t in rows]
        balance, as_of = repo.current_balance(session, user)
        series = detect_recurring(lite, as_of)
        usage = repo.save_recurring(session, uid, series)
        series = repo.recurring_candidates(session, uid)  # re-read: honours user flags (cancelled, usage)
        report = self._agent("analyzer").analyze(
            lite, series, as_of=as_of, currency=user.currency, usage_overrides=usage
        )
        # persist: anomalies & impulse tags on transactions, findings as insights
        session.execute(
            update(Transaction)
            .where(Transaction.user_id == uid)
            .values(is_anomaly=False, anomaly_reason=None, tags=None)
        )
        by_id = {t.id: t for t in rows}
        for a in report.anomalies:
            if a.txn_id in by_id:
                by_id[a.txn_id].is_anomaly, by_id[a.txn_id].anomaly_reason = True, a.reason
        for tid, tags in report.impulse.get("tagged", {}).items():
            if tid in by_id:
                by_id[tid].tags = ["impulse", *tags]
        for f in report.findings:
            session.add(
                Insight(
                    user_id=uid,
                    run_id=state["run_id"],
                    kind=f.kind,
                    severity=f.severity,
                    title=f.title,
                    detail=f.detail,
                    metrics=_jsonable(f.metrics),
                    transaction_ids=f.transaction_ids[:200],
                )
            )
        run = session.get(AgentRun, state["run_id"])
        if run is not None:
            snap = report.to_dict()
            snap["impulse"] = {k: v for k, v in snap["impulse"].items() if k != "tagged"}
            snap.pop("findings", None)
            run.analysis = _jsonable(snap)
        session.commit()
        kinds: dict[str, int] = {}
        for f in report.findings:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        return (
            {"analysis": report},
            f"{len(series)} recurring series, {len(report.subscriptions)} subscriptions, "
            f"{len(report.anomalies)} anomalies, {len(report.findings)} findings",
            {
                "findings_by_kind": kinds,
                "profile": report.profile,
                "profile_model": report.profile_model,
                "recurring": [s.merchant for s in series],
                "top_findings": [f.title for f in report.findings[:6]],
            },
        )

    def _forecast(self, session: Session, state: PipelineState):
        uid = state["user_id"]
        user = repo.get_or_create_user(session, uid)
        rows = repo.load_transactions(session, uid)
        lite = [repo.to_lite(t) for t in rows]
        balance, as_of = repo.current_balance(session, user)
        series = repo.recurring_candidates(session, uid)
        typed = ensure_goal_types(session, uid, as_of, user.currency, jev=self.jev, use_jev=self.use_jev)
        goals = repo.goal_inputs(session, uid)
        agent = self._agent(
            "forecaster",
            n_paths=self.settings.monte_carlo_paths,
            horizon_days=self.settings.forecast_horizon_days,
        )
        eng = agent.engine(
            lite,
            series,
            balance=balance,
            as_of=as_of,
            goals=goals,
            exclude_ids={t.id for t in rows if t.is_anomaly},
            safety_buffer=user.safety_buffer or self.settings.safety_buffer,
        )
        result = agent.forecast(eng, currency=user.currency)
        history = repo.goal_history(session, uid)  # before this run's snapshot → used for adaptation
        session.add(
            Forecast(user_id=uid, run_id=state["run_id"], horizon_days=eng.H, payload=_jsonable(result))
        )
        repo.append_goal_snapshots(session, uid, result["goals"], state["run_id"])
        session.commit()
        cp = {c["days"]: c for c in result["checkpoints"]}.get(90)
        summ = f"Balance {balance:,.0f} → ~{cp['p50']:,.0f} in 90 days (P50)" if cp else "Forecast ready"
        risky = [
            g["name"] for g in result["goals"] if g["status"] != "on_track" and g["status"] != "achieved"
        ]
        if risky:
            summ += f"; goals at risk: {', '.join(risky)}"
        result["_goal_history"] = history
        return (
            {"forecast": result},
            summ,
            {
                "horizon_days": eng.H,
                "paths": eng.n_paths,
                "safety_buffer": eng.safety_buffer,
                "low_point": result["low_point"],
                "prob_overdraft": result["prob_overdraft"],
                "goals": [
                    {k: g[k] for k in ("name", "probability", "status", "shortfall", "spend_at_deadline")}
                    for g in result["goals"]
                ],
                "goals_typed": typed,
                "narrative_model": result.get("narrative_model"),
            },
        )

    def _advise(self, session: Session, state: PipelineState):
        uid = state["user_id"]
        user = repo.get_or_create_user(session, uid)
        report = state["analysis"]
        forecast = dict(state.get("forecast") or {})
        history = forecast.pop("_goal_history", {})
        agent = self._agent("advisor")
        out = agent.advise(
            report,
            forecast,
            currency=user.currency,
            goal_history=history,
            feedback=repo.recommendation_feedback(session, uid),
        )
        for r in out["recommendations"]:
            session.add(
                Recommendation(
                    user_id=uid,
                    run_id=state["run_id"],
                    kind=r["kind"],
                    title=r["title"],
                    message=r["message"],
                    action=r["action"],
                    monthly_impact=r["monthly_impact"] + r["one_time_impact"] / 6,
                    priority=r["priority"],
                    urgency=r["urgency"],
                    status=r["status"],
                    trigger=r["trigger"],
                    evidence=_jsonable(
                        dict(r["evidence"], one_time_impact=r["one_time_impact"], confidence=r["confidence"])
                    ),
                    reasoning=r["reasoning"],
                    generated_by=r["generated_by"],
                    grounded=r["grounded"],
                    fingerprint=r["fingerprint"],
                )
            )
        session.commit()
        return (
            {"advice": out},
            f"{len(out['recommendations'])} recommendations (of {out['considered']} candidates, "
            f"{out['suppressed']} snoozed by your feedback); headline: {out['headline']}",
            {
                "headline": out["headline"],
                "model": out["model"],
                "recommendations": [
                    {"title": r["title"], "trigger": r["trigger"], "priority": r["priority"]}
                    for r in out["recommendations"]
                ],
            },
        )

    def _digest(self, session: Session, state: PipelineState):
        uid = state["user_id"]
        user = repo.get_or_create_user(session, uid)
        report: AnalysisReport = state["analysis"]
        advice = state.get("advice") or {}
        findings = [
            {"title": f.title, "detail": f.detail, "severity": f.severity, "kind": f.kind}
            for f in report.findings
            if f.kind != "subscription_inventory"
        ]
        recs = advice.get("recommendations", [])
        d = self._agent("digest").compose(
            user_name=user.name,
            currency=user.currency,
            findings=findings,
            recommendations=recs,
            forecast=state.get("forecast"),
            headline=advice.get("headline"),
        )
        deliveries = notifier.deliver(d, email_to=user.digest_email, telegram_chat_id=user.telegram_chat_id)
        rec_row = session.scalar(
            select(Recommendation)
            .where(Recommendation.run_id == state["run_id"])
            .order_by(Recommendation.priority.desc())
            .limit(1)
        )
        row = Digest(
            user_id=uid,
            run_id=state["run_id"],
            trigger=state.get("trigger", "manual"),
            subject=d["subject"],
            headline=d["headline"],
            action=d["action"],
            body_markdown=d["body_markdown"],
            deliveries=deliveries,
            recommendation_id=rec_row.id if rec_row else None,
        )
        session.add(row)
        session.commit()
        sent = [k for k, v in deliveries.items() if v["status"] == "sent"]
        return (
            {"digest_id": row.id},
            f"Digest '{d['subject']}' delivered via {', '.join(sent)}",
            {"deliveries": deliveries, "model": d["model"]},
        )

    @staticmethod
    def _summary(state: PipelineState) -> dict[str, Any]:
        f = state.get("forecast") or {}
        a: AnalysisReport | None = state.get("analysis")
        adv = state.get("advice") or {}
        return _jsonable(
            {
                "plan": state.get("plan"),
                "transactions": state.get("n_transactions"),
                "categorization": state.get("categorization"),
                "findings": len(a.findings) if a else 0,
                "anomalies": len(a.anomalies) if a else 0,
                "recommendations": len(adv.get("recommendations", [])),
                "headline": adv.get("headline"),
                "low_point": f.get("low_point"),
                "digest_id": state.get("digest_id"),
            }
        )


def _summarise_jev(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Jev calls can number in the hundreds per step; keep the totals, not every row."""
    by_purpose: dict[str, int] = {}
    for c in calls:
        by_purpose[c.get("purpose") or "?"] = by_purpose.get(c.get("purpose") or "?", 0) + 1
    return {
        "count": len(calls),
        "cached": sum(1 for c in calls if c.get("cached")),
        "failed": sum(1 for c in calls if not c.get("success")),
        "input_tokens": sum(int(c.get("prompt_tokens") or 0) for c in calls),
        "cost_usd": round(sum(float(c.get("cost_usd") or 0) for c in calls), 6),
        "latency_ms_max": max((int(c.get("latency_ms") or 0) for c in calls), default=0),
        "by_purpose": by_purpose,
        "model": next((f"{c['provider']}:{c['model']}" for c in calls if c.get("model")), None),
    }


def _jsonable(obj: Any) -> Any:
    """Make nested structures JSON-safe (dates → ISO strings, numpy → float, sets → lists)."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
