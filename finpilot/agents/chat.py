"""Conversational agent — answers free-form money questions from the user's real data.

With an LLM: a tool-calling loop (OpenAI function-calling via Mesh) over `FinanceToolkit`.
Without one: a deterministic intent router that calls the *same* tools, so "Can I afford a
₹40k trip next month?" is still answered with a real simulation.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Any

from finpilot.agents.base import BaseAgent
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


class ChatAgent(BaseAgent):
    name = "chat"
    role = "Answers free-form money questions with tool calls over the user's real data"
    system_prompt = SYSTEM_PROMPT

    def answer(
        self, kit: FinanceToolkit, question: str, history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        if self.llm_available:
            try:
                return self._llm_answer(kit, question, history or [])
            except (LLMError, ValueError) as exc:
                log.warning("chat LLM failed, falling back to rules: %s", exc)
        return self._rule_answer(kit, question)

    # ------------------------------------------------------------------ LLM tool loop
    def _llm_answer(
        self, kit: FinanceToolkit, question: str, history: list[dict[str, str]]
    ) -> dict[str, Any]:
        system = SYSTEM_PROMPT.format(as_of=kit.as_of.isoformat(), currency=kit.currency)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for h in history[-8:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})
        trace: list[dict[str, Any]] = []
        model = None
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.router.chat(
                messages, agent=self.name, tools=TOOL_SPECS, temperature=0.2, max_tokens=700, cache=False
            )
            model = resp.label
            if not resp.tool_calls:
                return {
                    "answer": (resp.content or "").strip() or "I couldn't produce an answer.",
                    "trace": trace,
                    "model": model,
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
        )
        return {"answer": (resp.content or "").strip(), "trace": trace, "model": resp.label or model}

    # ------------------------------------------------------------------ deterministic fallback
    def _rule_answer(self, kit: FinanceToolkit, question: str) -> dict[str, Any]:
        q = question.lower()
        trace: list[dict[str, Any]] = []

        def call(name: str, **args: Any) -> dict[str, Any]:
            res = kit.call(name, args)
            trace.append({"tool": name, "args": args, "result": _preview(res)})
            return res

        money = kit.money
        if re.search(
            r"\b(afford|can i (buy|spend|go|take|get)|should i (buy|spend|book)|what if i (buy|spend))", q
        ):
            amount = parse_amount(question)
            when = parse_date(question, kit.as_of)
            note = ""
            is_trip = bool(re.search(r"trip|vacation|holiday|travel|goa|getaway|flight", q))
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
            item = "trip" if is_trip else "purchase"
            sim = call(
                "simulate",
                adjustments=[
                    {
                        "type": "one_time",
                        "amount": amount,
                        "date": when.isoformat(),
                        "label": item.capitalize(),
                    }
                ],
            )
            head = {
                "affordable": "**Yes — you can afford it.**",
                "stretch": "**It's a stretch.**",
                "not_affordable": "**Not right now.**",
            }[sim["verdict"]]
            lines = [
                note
                + f"{head} I simulated a {money(amount)} {item} on {when:%d %b %Y} against your forecast.",
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

        if re.search(r"how much|spent|spend|spending|expense", q) and not re.search(r"save|cut|reduce", q):
            cat_word = next(
                (w for w in sorted(SYNONYMS, key=len, reverse=True) if re.search(rf"\b{w}\b", q)), None
            )
            cats = resolve_categories(cat_word)
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
            if not cats:
                m = re.search(r"\b(?:on|at)\s+([a-z0-9.&' ]{3,30}?)(?:\s+(?:this|last|in)\b|\?|$)", q)
                merchant = m.group(1).strip() if m else None
            res = call("get_spending", category=cat_word if cats else None, merchant=merchant, period=period)
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
                lines.append(
                    "By month: " + ", ".join(f"{k} {money(v)}" for k, v in res["monthly"].items()) + "."
                )
            return {"answer": "\n\n".join(lines), "trace": trace, "model": "rules"}

        if "subscri" in q:
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

        if re.search(r"goal|on track|emergency fund|saving for", q):
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
                    f" — {money(g['shortfall'])} short at this pace; {money(g['required_monthly'])}/mo more "
                    f"closes it"
                    if g.get("shortfall")
                    else ""
                )
                lines.append(
                    f"- **{g['name']}**: {status} ({g['probability'] * 100:.0f}% likely by "
                    f"{g['deadline']}{eta}){gap}"
                )
            return {"answer": "\n".join(lines), "trace": trace, "model": "rules"}

        if re.search(
            r"forecast|balance|heading|trajectory|runway|end of|in \d+ (days|weeks|months)|next month", q
        ):
            target = (
                parse_date(question, kit.as_of) if re.search(r"in \d+|next|by|end of|\d{4}-", q) else None
            )
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

        if re.search(r"save|cut|reduce|advice|recommend|tip|improve|what should i", q):
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

        if re.search(r"unusual|anomal|weird|suspicious|duplicate|strange", q):
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

        ov = call("get_overview")
        l30 = ov["last_30_days"]
        cats = ", ".join(f"{c['category']} {money(c['amount'])}" for c in ov["top_categories_30d"][:3])
        rate = f"{l30['savings_rate_pct']:.0f}%" if l30["savings_rate_pct"] is not None else "n/a"
        ans = (
            f"Balance **{money(ov['balance'])}** as of {ov['as_of']}. Last 30 days: income {money(l30['income'])}, "
            f"spending {money(l30['spend'])} (savings rate {rate}). Biggest categories: {cats}.\n\n"
            "Try asking: *“Can I afford a trip next month?”*, *“How much did I spend on food delivery last "
            "month?”*, *“Am I on track for my emergency fund?”*, *“What are my subscriptions?”*"
        )
        return {"answer": ans, "trace": trace, "model": "rules"}


def _preview(result: dict[str, Any]) -> Any:
    text = json.dumps(result, default=str)
    if len(text) <= 1500:
        return result
    return {"_truncated": True, "summary": text[:1500]}
