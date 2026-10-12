"""Conversational agent — answers free-form money questions from the user's real data.

Two systems, one toolkit (`FinanceToolkit`):

* **System One — Jev routes every message** in a single call: intent (12 options, including
  off-topic, prompt-injection and "please move my money"), the category / period / purchase
  slots, and a complexity score. Confident, simple intents are answered by deterministic
  handlers that call the tools directly — no LLM tokens at all.
* **System Two — the LLM tool loop** (OpenAI function-calling via Mesh) handles what is left:
  questions Jev marks as complex or is unsure about. The complexity score picks the model
  (`CHAT_FAST_MODEL` for simpler escalations), and the answer goes through the faithfulness guard.

Amounts and dates are parsed in code (Jev is not a calculator). Without Jev, a regex intent
router does the routing; without an LLM, the handlers answer everything.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from finpilot.agents.base import BaseAgent
from finpilot.decisions.guard import check_faithfulness
from finpilot.decisions.jev import Evaluation, JevError
from finpilot.decisions.policy import log_decisions, order_debias, record, top_probabilities
from finpilot.decisions.spec import chat_questions, chat_state
from finpilot.domain.stats import add_months
from finpilot.llm.client import LLMError
from finpilot.services.toolkit import SYNONYMS, TOOL_SPECS, FinanceToolkit, resolve_categories

log = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You are FinPilot, a proactive personal-finance co-pilot. You answer questions about ONE user's money
using tools that read their real transactions, forecast and goals. Today (latest data) is {as_of}. Currency: {currency}.

Rules:
- Every number you state must come from a tool result in this conversation. Never guess or invent figures.
- For "can I afford / should I buy / what if" questions, call `simulate` with the right adjustments. If the user gives
  no amount for a trip, call `estimate_trip_cost` first and say it is an estimate based on their past trips.
  "Next month" means a date about 30 days after {as_of}.
- Lead with a direct answer (Yes / No / It's a stretch), then 2-3 short bullet reasons with numbers,
  then ONE concrete next step. Keep it under 130 words. Use markdown.
- Be specific to this user's patterns; no generic advice."""

AMOUNT_RE = re.compile(
    r"(?:₹|rs\.?|inr|\$)?\s*(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)\s*(k|thousand|l|lakh|lakhs|lac|cr|crore)?\b",
    re.IGNORECASE,
)
MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})


def parse_amount(text: str) -> float | None:
    best = None
    for m in AMOUNT_RE.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        prefix = text[max(0, m.start() - 1) : m.start() + 1]
        if not unit and "₹" not in m.group(0) and "rs" not in m.group(0).lower() and "$" not in prefix:
            # bare numbers: only accept if large enough to be money (avoid "3 months", "2026")
            v = float(raw.replace(",", ""))
            if v < 500 or (1900 <= v <= 2100 and "," not in raw):
                continue
        v = float(raw.replace(",", ""))
        v *= {
            "k": 1e3,
            "thousand": 1e3,
            "l": 1e5,
            "lakh": 1e5,
            "lakhs": 1e5,
            "lac": 1e5,
            "cr": 1e7,
            "crore": 1e7,
        }.get(unit, 1)
        best = max(best or 0, v)
    return best


def parse_date(text: str, as_of: dt.date) -> dt.date:
    t = text.lower()
    if "tomorrow" in t:
        return as_of + dt.timedelta(days=1)
    if "next week" in t:
        return as_of + dt.timedelta(days=7)
    if "next month" in t:
        return add_months(as_of, 1)
    if "next year" in t:
        return add_months(as_of, 12)
    m = re.search(r"in (\d+) (day|week|month)s?", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return (
            add_months(as_of, n)
            if unit == "month"
            else as_of + dt.timedelta(days=n * (7 if unit == "week" else 1))
        )
    m = re.search(r"(\d{4}-\d{2}-\d{2})", t)
    if m:
        return dt.date.fromisoformat(m.group(1))
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", t):
            year = as_of.year + (1 if num < as_of.month or (num == as_of.month and as_of.day > 20) else 0)
            return dt.date(year, num, 15)
    return as_of + dt.timedelta(days=14)


Handler = Callable[[FinanceToolkit, str, dict[str, Any]], dict[str, Any]]

# Intents a deterministic handler answers fully; the rest go to the LLM (or are refused).
DIRECT_INTENTS = (
    "afford_check",
    "spending_lookup",
    "subscriptions",
    "goal_status",
    "forecast",
    "savings_advice",
    "anomalies",
    "overview",
)
BLOCKED_INTENTS = ("action_request", "off_topic", "prompt_injection")
EXAMPLES = (
    "Try asking: *“Can I afford a trip next month?”*, *“How much did I spend on food delivery last "
    "month?”*, *“Am I on track for my emergency fund?”*, *“What are my subscriptions?”*"
)
REFUSALS = {
    "action_request": "I can't move money, pay bills or change subscriptions — FinPilot is read-only, so your "
    "accounts stay under your control. I can tell you what an action would do first, e.g. *“What happens if "
    "I cancel Netflix?”* or *“Can I afford to pay ₹20,000 to Rahul this week?”*",
    "off_topic": "I only answer questions about your own money — spending, subscriptions, goals and your "
    "forecast. " + EXAMPLES,
    "prompt_injection": "I can't change how I work or share internal instructions. I'm happy to help with your "
    "finances, though. " + EXAMPLES,
}


@dataclass
class Route:
    """System One's routing decision for one chat message."""

    intent: str
    confidence: float
    probabilities: dict[str, float]
    slots: dict[str, Any]
    complexity: float
    complexity_label: str
    model: str
    input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False
    order_checked: bool = False
    action: str = "pending"  # answered_by_rules | escalated_to_llm | refused | fallback_rules
    llm_model: str | None = None
    evaluation: Evaluation | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 3),
            "probabilities": top_probabilities(self.probabilities, 3),
            "slots": self.slots,
            "complexity": round(self.complexity, 2),
            "complexity_label": self.complexity_label,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "cost_usd": round(self.cost_usd, 7),
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "order_checked": self.order_checked,
            "action": self.action,
            "llm_model": self.llm_model,
        }


class ChatAgent(BaseAgent):
    name = "chat"
    role = "Routes each question with Jev, answers from tools; escalates only complex ones to the LLM"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.handlers: dict[str, Handler] = {
            "afford_check": self._h_afford,
            "spending_lookup": self._h_spending,
            "subscriptions": self._h_subscriptions,
            "goal_status": self._h_goals,
            "forecast": self._h_forecast,
            "savings_advice": self._h_advice,
            "anomalies": self._h_anomalies,
            "overview": self._h_overview,
        }

    # ------------------------------------------------------------------ entry point
    def answer(
        self, kit: FinanceToolkit, question: str, history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        history = history or []
        route = self.route(question, history)
        if route is not None:
            if route.intent in BLOCKED_INTENTS and route.confidence >= self.jev.route_confidence:
                return self._finish(
                    route,
                    {"answer": REFUSALS[route.intent], "trace": [], "model": route.model},
                    "refused",
                    question,
                )
            if route.intent in DIRECT_INTENTS and route.confidence >= self.jev.route_confidence:
                res = self.handlers[route.intent](kit, question, route.slots)
                return self._finish(route, res, "answered_by_rules", question)
        if self.llm_available:
            model = self._pick_model(route)
            try:
                res = self._llm_answer(kit, question, history, model=model)
                res["answer"] = self._guard(res)
                if route is not None:
                    route.llm_model = res["model"]
                return self._finish(route, res, "escalated_to_llm", question)
            except (LLMError, ValueError) as exc:
                log.warning("chat LLM failed, falling back to rules: %s", exc)
        intent = self._regex_intent(question)
        if (
            route is not None
            and route.intent in DIRECT_INTENTS
            and route.confidence >= self.jev.accept_confidence
            and intent == "overview"
        ):
            intent = route.intent  # regex found nothing specific; Jev's best guess is better than a summary
        res = self.handlers[intent](kit, question, route.slots if route else {})
        return self._finish(route, res, "fallback_rules", question)

    # ------------------------------------------------------------------ System One routing
    def route(self, question: str, history: list[dict[str, str]] | None = None) -> Route | None:
        if not self.jev_available:
            return None
        state, qs = chat_state(question, history), chat_questions()
        try:
            ev = self.jev.evaluate(state, qs, agent=self.name, purpose="chat_route")
        except JevError as exc:
            log.warning("Jev routing failed, using the fallback router: %s", exc)
            return None
        ev = order_debias(self.jev, state, qs, ev, agent=self.name, purpose="chat_route", names={"intent"})
        slot_min = self.jev.accept_confidence
        cat, period, kind = ev["spend_category"], ev["period"], ev["purchase_kind"]
        slots = {
            "category": cat.choice if cat.choice != "none" and cat.confidence >= slot_min else None,
            "period": period.choice if period.confidence >= slot_min else None,
            "purchase_kind": kind.choice if kind.choice != "other" and kind.confidence >= slot_min else None,
        }
        intent, cx = ev["intent"], ev["complexity"]
        return Route(
            intent=intent.choice,
            confidence=intent.confidence,
            probabilities=intent.probabilities,
            slots=slots,
            complexity=cx.score,
            complexity_label=cx.label,
            model=ev.label,
            input_tokens=ev.input_tokens,
            cost_usd=ev.cost_usd,
            latency_ms=ev.latency_ms,
            cached=ev.cached,
            order_checked="intent" in ev.meta.get("order_checked", []),
            evaluation=ev,
        )

    def _pick_model(self, route: Route | None) -> str | None:
        """Cheap model for escalations Jev scores as simple; the configured chat model otherwise."""
        fast = self.router.settings.chat_fast_model
        if fast and route is not None and route.complexity < 1.5:
            return fast
        return None

    def _finish(self, route: Route | None, res: dict[str, Any], action: str, question: str) -> dict[str, Any]:
        if route is None:
            return {**res, "route": None}
        route.action = action
        if route.evaluation is not None:
            log_decisions(
                [
                    record(
                        route.evaluation,
                        "intent",
                        agent=self.name,
                        purpose="chat_route",
                        subject=question,
                        action={
                            "answered_by_rules": "accepted",
                            "refused": "accepted",
                            "escalated_to_llm": "escalated",
                        }.get(action, "deferred"),
                    )
                ]
            )
        return {**res, "route": route.to_dict()}

    def _guard(self, res: dict[str, Any]) -> str:
        """Faithfulness check of an LLM answer against the tool results it was built from."""
        answer = res["answer"]
        if not res["trace"]:
            return answer
        facts = [{"tool": t["tool"], "result": t["result"]} for t in res["trace"]]
        (verdict,) = check_faithfulness(self.jev, [("chat answer", facts, answer)], agent=self.name)
        res["guard"] = {"checked": verdict.checked, "ok": verdict.ok, "label": verdict.label}
        if verdict.checked and not verdict.ok:
            return (
                answer
                + "\n\n_⚠️ Some statements above could not be verified against your data — check the tool "
                "results below._"
            )
        return answer

    # ------------------------------------------------------------------ System Two: LLM tool loop
    def _llm_answer(
        self,
        kit: FinanceToolkit,
        question: str,
        history: list[dict[str, str]],
        model: str | None = None,
    ) -> dict[str, Any]:
        system = SYSTEM_PROMPT.format(as_of=kit.as_of.isoformat(), currency=kit.currency)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for h in history[-8:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})
        trace: list[dict[str, Any]] = []
        label_ = None
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.router.chat(
                messages,
                agent=self.name,
                tools=TOOL_SPECS,
                temperature=0.2,
                max_tokens=700,
                cache=False,
                model=model,
            )
            label_ = resp.label
            if not resp.tool_calls:
                return {
                    "answer": (resp.content or "").strip() or "I couldn't produce an answer.",
                    "trace": trace,
                    "model": label_,
                }
            messages.append(resp.assistant_message())
            for tc in resp.tool_calls:
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = kit.call(tc["name"], args)
                trace.append({"tool": tc["name"], "args": args, "result": _preview(result)})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, default=str)[:6000],
                    }
                )
        # ran out of rounds: ask for a final answer without tools
        resp = self.router.chat(
            messages + [{"role": "user", "content": "Answer now using the tool results above."}],
            agent=self.name,
            temperature=0.2,
            max_tokens=500,
            cache=False,
            model=model,
        )
        return {"answer": (resp.content or "").strip(), "trace": trace, "model": resp.label or label_}

    # ------------------------------------------------------------------ fallback router (no Jev)
    @staticmethod
    def _regex_intent(question: str) -> str:
        q = question.lower()
        if re.search(
            r"\b(afford|can i (buy|spend|go|take|get)|should i (buy|spend|book)|what if i (buy|spend))", q
        ):
            return "afford_check"
        if re.search(r"how much|spent|spend|spending|expense", q) and not re.search(r"save|cut|reduce", q):
            return "spending_lookup"
        if "subscri" in q:
            return "subscriptions"
        if re.search(r"goal|on track|emergency fund|saving for", q):
            return "goal_status"
        if re.search(
            r"forecast|balance|heading|trajectory|runway|end of|in \d+ (days|weeks|months)|next month", q
        ):
            return "forecast"
        if re.search(r"save|cut|reduce|advice|recommend|tip|improve|what should i", q):
            return "savings_advice"
        if re.search(r"unusual|anomal|weird|suspicious|duplicate|strange", q):
            return "anomalies"
        return "overview"

    def _rule_answer(self, kit: FinanceToolkit, question: str) -> dict[str, Any]:
        """Answer with the regex router only (kept for callers that bypass System One)."""
        return self.handlers[self._regex_intent(question)](kit, question, {})

    # ------------------------------------------------------------------ deterministic handlers
    @staticmethod
    def _caller(kit: FinanceToolkit, trace: list[dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        def call(name: str, **args: Any) -> dict[str, Any]:
            res = kit.call(name, args)
            trace.append({"tool": name, "args": args, "result": _preview(res)})
            return res

        return call

    def _h_afford(self, kit: FinanceToolkit, question: str, slots: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        call, money, q = self._caller(kit, trace), kit.money, question.lower()
        amount = parse_amount(question)
        when = parse_date(question, kit.as_of)
        note = ""
        kind = slots.get("purchase_kind")
        is_trip = (
            kind == "trip" if kind else bool(re.search(r"trip|vacation|holiday|travel|goa|getaway|flight", q))
        )
        if amount is None and is_trip:
            est = call("estimate_trip_cost")
            amount = est["estimate"]
            note = f"_I estimated the trip at {money(amount)} ({est['basis']})._\n\n"
        if amount is None:
            return {
                "answer": "How much would it cost? Tell me the amount (e.g. *“Can I afford a ₹45,000 laptop "
                "next month?”*) and I'll simulate it against your real trajectory.",
                "trace": trace,
                "model": "rules",
            }
        item = "trip" if is_trip else ("event" if kind == "event" else "purchase")
        sim = call(
            "simulate",
            adjustments=[
                {"type": "one_time", "amount": amount, "date": when.isoformat(), "label": item.capitalize()}
            ],
        )
        head = {
            "affordable": "**Yes — you can afford it.**",
            "stretch": "**It's a stretch.**",
            "not_affordable": "**Not right now.**",
        }[sim["verdict"]]
        lines = [
            note + f"{head} I simulated a {money(amount)} {item} on {when:%d %b %Y} against your forecast.",
            "",
        ]
        lines += [f"- {r}" for r in sim["reasons"]]
        for g in sim["goals"]:
            if g["prob_before"] != g["prob_after"]:
                lines.append(
                    f"- '{g['name']}': {g['prob_before'] * 100:.0f}% → {g['prob_after'] * 100:.0f}% "
                    f"chance of hitting it on time."
                )
        if sim["suggestions"]:
            lines += ["", "**Next step:** " + sim["suggestions"][0]]
        return {"answer": "\n".join(lines), "trace": trace, "model": "rules"}

    def _h_spending(self, kit: FinanceToolkit, question: str, slots: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        call, money, q = self._caller(kit, trace), kit.money, question.lower()
        category = slots.get("category")
        if category is None:
            word = next(
                (w for w in sorted(SYNONYMS, key=len, reverse=True) if re.search(rf"\b{w}\b", q)), None
            )
            category = word if resolve_categories(word) else None
        period = slots.get("period")
        if period is None:
            period = "last_30_days"
            if "last month" in q:
                period = "last_month"
            elif "this month" in q:
                period = "this_month"
            elif re.search(r"90 days|3 months|quarter", q):
                period = "last_90_days"
            elif "this year" in q:
                period = "this_year"
        merchant = None
        if not category:
            m = re.search(r"\b(?:on|at|to)\s+([a-z0-9.&' ]{3,30}?)(?:\s+(?:this|last|in)\b|\?|$)", q)
            merchant = m.group(1).strip() if m else None
        res = call("get_spending", category=category, merchant=merchant, period=period)
        what = ", ".join(res["filter"]["categories"]) or (
            merchant or "everything (excl. transfers & investments)"
        )
        when = f"the {res['period']}" if res["period"].startswith("last") else res["period"]
        lines = [
            f"You spent **{money(res['total'])}** on {what} in {when} "
            f"({res['count']} transactions, avg {money(res['avg_per_txn'])})."
        ]
        if res["top_merchants"]:
            lines.append(
                "Top merchants: "
                + ", ".join(f"{m['merchant']} {money(m['amount'])}" for m in res["top_merchants"][:3])
                + "."
            )
        if len(res["monthly"]) > 1:
            lines.append("By month: " + ", ".join(f"{k} {money(v)}" for k, v in res["monthly"].items()) + ".")
        return {"answer": "\n\n".join(lines), "trace": trace, "model": "rules"}

    def _h_subscriptions(self, kit: FinanceToolkit, question: str, slots: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        call, money = self._caller(kit, trace), kit.money
        res = call("get_subscriptions")
        lines = [
            f"You have **{res['count_active']} active subscriptions** costing "
            f"**{money(res['monthly_total'])}/month**:"
        ]
        for i in res["items"]:
            if i["status"] != "active":
                continue
            flag = (
                f" — {i['usage_signal']}"
                if i.get("usage_signal") and "No " in (i["usage_signal"] or "")
                else ""
            )
            hike = f" (price +{i['price_change_pct']:.0f}%)" if i.get("price_change_pct") else ""
            lines.append(f"- {i['merchant']}: {money(i['monthly_cost'])}/mo{hike}{flag}")
        return {"answer": "\n".join(lines), "trace": trace, "model": "rules"}

    def _h_goals(self, kit: FinanceToolkit, question: str, slots: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        call, money = self._caller(kit, trace), kit.money
        res = call("get_goals")
        if not res["goals"]:
            return {
                "answer": "You haven't set any goals yet — add one on the Goals page and I'll track it.",
                "trace": trace,
                "model": "rules",
            }
        lines = []
        for g in res["goals"]:
            status = g["status"].replace("_", " ")
            eta = f", ETA {g['eta']}" if g.get("eta") else ""
            gap = (
                f" — {money(g['shortfall'])} short at this pace; {money(g['required_monthly'])}/mo more closes it"
                if g.get("shortfall")
                else ""
            )
            lines.append(
                f"- **{g['name']}**: {status} ({g['probability'] * 100:.0f}% likely by {g['deadline']}{eta}){gap}"
            )
        return {"answer": "\n".join(lines), "trace": trace, "model": "rules"}

    def _h_forecast(self, kit: FinanceToolkit, question: str, slots: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        call, money, q = self._caller(kit, trace), kit.money, question.lower()
        target = parse_date(question, kit.as_of) if re.search(r"in \d+|next|by|end of|\d{4}-", q) else None
        res = call("get_forecast", date=target.isoformat() if target else None)
        lines = [f"Your balance today is **{money(res['start_balance'])}**."]
        if res["at_date"]:
            a = res["at_date"]
            lines.append(
                f"By {a['date']} it's projected at **{money(a['p50'])}** "
                f"(likely range {money(a['p10'])}–{money(a['p90'])})."
            )
        for c in res["checkpoints"][:3]:
            lines.append(f"- In {c['days']} days: ~{money(c['p50'])}")
        lp = res["low_point"]
        lines.append(f"Tightest point: ~{money(lp['p50'])} around {lp['date']}.")
        return {"answer": "\n".join(lines), "trace": trace, "model": "rules"}

    def _h_advice(self, kit: FinanceToolkit, question: str, slots: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        call, money = self._caller(kit, trace), kit.money
        res = call("get_recommendations")
        if not res["recommendations"]:
            return {
                "answer": "No recommendations yet — run the analysis first.",
                "trace": trace,
                "model": "rules",
            }
        lines = ["Here's what would make the biggest difference for you:"]
        for r in res["recommendations"][:4]:
            impact = f" (~{money(r['monthly_impact'])}/mo)" if r["monthly_impact"] else ""
            lines.append(f"- **{r['title']}**{impact} — {r['message']}")
        return {"answer": "\n".join(lines), "trace": trace, "model": "rules"}

    def _h_anomalies(self, kit: FinanceToolkit, question: str, slots: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        call = self._caller(kit, trace)
        res = call("get_insights")
        items = [i for i in res["insights"] if i["kind"].startswith("anomaly")]
        if not items:
            return {
                "answer": "Nothing unusual in your recent transactions.",
                "trace": trace,
                "model": "rules",
            }
        return {
            "answer": "\n".join(f"- **{i['title']}** — {i['detail']}" for i in items),
            "trace": trace,
            "model": "rules",
        }

    def _h_overview(self, kit: FinanceToolkit, question: str, slots: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        call, money = self._caller(kit, trace), kit.money
        ov = call("get_overview")
        l30 = ov["last_30_days"]
        cats = ", ".join(f"{c['category']} {money(c['amount'])}" for c in ov["top_categories_30d"][:3])
        rate = f"{l30['savings_rate_pct']:.0f}%" if l30["savings_rate_pct"] is not None else "n/a"
        ans = (
            f"Balance **{money(ov['balance'])}** as of {ov['as_of']}. Last 30 days: income {money(l30['income'])}, "
            f"spending {money(l30['spend'])} (savings rate {rate}). Biggest categories: {cats}.\n\n"
            + EXAMPLES
        )
        return {"answer": ans, "trace": trace, "model": "rules"}


def _preview(result: dict[str, Any]) -> Any:
    text = json.dumps(result, default=str)
    if len(text) <= 1500:
        return result
    return {"_truncated": True, "summary": text[:1500]}
