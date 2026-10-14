"""Digest agent — composes the proactive weekly digest: one key insight + one action."""

from __future__ import annotations

import html
from typing import Any

from finpilot.agents.base import BaseAgent
from finpilot.llm.grounding import check_grounding

SYSTEM_PROMPT = """You are the Digest agent of FinPilot. Write the opening of a short weekly money email to ONE user.
You get their single most important insight, the single action we recommend, and a few outlook numbers.
Write exactly 2 sentences: (1) a warm, specific opener that states the insight with its key figure; (2) why the
action matters now and what it changes. Preserve missing-data caveats and conditional savings; net cash-flow changes
are not balances. Never say "all steady" or "no action needed" just because no recommendation was found.
Copy figures exactly as given (keep decimals); never round or invent numbers. No "Dear user", no sign-off.
Also write an email subject (max 60 characters) naming the specific merchant, category or goal. Currency: {currency}.
Respond with JSON only: {{"intro": "...", "subject": "..."}}"""


def _money(x: float, currency: str) -> str:
    sym = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")
    return f"{'-' if x < 0 else ''}{sym}{abs(x):,.0f}"


class DigestAgent(BaseAgent):
    name = "digest"
    role = "Composes the scheduled digest (key insight + one action) for email / Telegram / in-app"
    system_prompt = SYSTEM_PROMPT

    def compose(
        self,
        *,
        user_name: str,
        currency: str,
        findings: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        forecast: dict[str, Any] | None,
        headline: str | None = None,
    ) -> dict[str, Any]:
        insight = next((f for f in findings if f["severity"] in ("critical", "warning")), None) or (
            findings[0] if findings else None
        )
        rec = recommendations[0] if recommendations else None
        insight_txt = (
            f"{insight['title']}. {insight['detail']}"
            if insight
            else "The available records did not support a specific new insight."
        )
        action_txt = (
            f"{rec['title']} — {rec['message']}"
            if rec
            else "Review data coverage and upcoming obligations; no supported action was identified."
        )
        subject = f"FinPilot weekly: {rec['title']}" if rec else "FinPilot weekly: review your records"
        lines = []
        if forecast:
            cps = {c["days"]: c for c in forecast.get("checkpoints", [])}
            cp = cps.get(30)
            if cp:
                lines.append(
                    f"{'Net cash-flow change' if forecast.get('projection_type') == 'cashflow_change' else 'Balance'} in 30 days: ~{_money(cp['p50'], currency)} "
                    f"(range {_money(cp['p10'], currency)}–{_money(cp['p90'], currency)})"
                )
            if forecast.get("projection_type") == "cashflow_change":
                lines.append("Starting balance is unverified; affordability and goal odds are unavailable.")
            for g in forecast.get("goals", [])[:3]:
                lines.append(f"{g['name']}: {g['probability'] * 100:.0f}% likely by {g['deadline']}")
        intro = headline or insight_txt
        model = "rules"
        if self.llm_available:
            facts = {
                "insight": insight,
                "action": rec and {k: rec[k] for k in ("title", "message", "monthly_impact")},
                "outlook": lines,
            }
            self.system_prompt = SYSTEM_PROMPT.format(currency=currency)
            data, label_, _ = self.safe_ask_json(facts, max_tokens=800, temperature=0.1)
            if (
                isinstance(data, dict)
                and data.get("intro")
                and check_grounding(str(data["intro"]), facts, strict=True)[0]
            ):
                draft_subject = str(data.get("subject") or "")
                subject_ok = bool(draft_subject) and check_grounding(draft_subject, facts, strict=True)[0]
                (verdict,) = self.validate(
                    [
                        (
                            "digest intro",
                            facts,
                            str(data["intro"]) + (f"\nSubject: {draft_subject}" if subject_ok else ""),
                        )
                    ],
                )
                if verdict.ok:
                    intro = str(data["intro"])
                    if subject_ok:
                        subject = draft_subject[:80]
                    model = label_ or "llm"
        others = [r for r in recommendations[1:4]]
        md = [
            f"Hi {user_name.split()[0] if user_name else 'there'},",
            "",
            intro,
            "",
            "**🔎 Key insight**",
            insight_txt,
            "",
            "**✅ Your one action this week**",
            action_txt,
        ]
        if rec and rec.get("monthly_impact"):
            md.append(
                f"_Conditional impact: about {_money(rec['monthly_impact'], currency)}/month if the proposed action and assumptions hold._"
            )
        if rec and rec.get("reasoning"):
            md += ["", f"_Why: {rec['reasoning'][0]}_"]
        if lines:
            md += ["", "**📈 Outlook**"] + [f"- {x}" for x in lines]
        if others:
            md += ["", "**Also worth a look**"] + [f"- {r['title']}" for r in others]
        md += ["", "— FinPilot"]
        body = "\n".join(md)
        return {
            "subject": subject,
            "headline": insight_txt,
            "action": action_txt,
            "body_markdown": body,
            "body_html": markdown_to_html(body),
            "model": model,
        }


def markdown_to_html(md: str) -> str:
    """Tiny markdown → HTML for emails (bold, italics, bullets, paragraphs)."""
    import re

    out, in_list = [], False
    for raw in md.split("\n"):
        line = html.escape(raw)
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", line)
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{line}</p>" if line.strip() else "")
    if in_list:
        out.append("</ul>")
    return (
        '<div style="font-family:Inter,Segoe UI,Arial,sans-serif;max-width:560px;line-height:1.5;color:#0f172a">'
        + "\n".join(out)
        + "</div>"
    )
