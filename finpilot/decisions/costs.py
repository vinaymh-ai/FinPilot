"""What System One saves — an explicit, inspectable cost model.

Jev cost is *measured* (input tokens on every call, from the `llm_calls` table). The LLM side is
an *estimate* of what the same decision cost before Jev, derived from FinPilot's own prompts:

* `merchant_category` — the batched LLM classifier: its system prompt (taxonomy) amortised over a
  40-merchant batch, plus ~70 input / ~45 output tokens per merchant line.
* `chat_route` — a chat answer that Jev routed to a deterministic handler would otherwise have run
  the LLM tool loop: ~2 rounds of (system prompt + tool schemas + question) + one tool result in,
  ~2 short completions out.

Only those two *replace* LLM calls, so only they count as savings. The faithfulness guard, goal
typing and CSV-column mapping are checks FinPilot did not have before Jev: they are reported as
added cost, never as savings.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from finpilot.config import Settings, get_settings
from finpilot.llm.client import llm_cost_usd

CHARS_PER_TOKEN = 4.0
REPLACES_LLM = {"merchant_category", "chat_route"}


def approx_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return int(len(text) / CHARS_PER_TOKEN) + 1


@lru_cache(maxsize=1)
def llm_equivalent_tokens() -> dict[str, tuple[int, int]]:
    """(input, output) tokens the pre-Jev LLM path spends per decision, by purpose."""
    from finpilot.agents.categorizer import SYSTEM_PROMPT as CAT_PROMPT
    from finpilot.agents.categorizer import _taxonomy_text
    from finpilot.agents.chat import SYSTEM_PROMPT as CHAT_PROMPT
    from finpilot.services.toolkit import TOOL_SPECS

    cat_system = approx_tokens(CAT_PROMPT.format(taxonomy=_taxonomy_text()))
    chat_round = approx_tokens(CHAT_PROMPT) + approx_tokens(TOOL_SPECS) + 30
    return {
        "merchant_category": (cat_system // 40 + 70, 45),
        "chat_route": (2 * chat_round + 1200, 2 * 120),
    }


def savings_report(
    jev_by_purpose: dict[str, dict[str, Any]],
    decisions_by_purpose: dict[str, dict[str, int]],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Combine measured Jev spend with the LLM-equivalent estimate.

    `jev_by_purpose`        {purpose: {"calls", "input_tokens", "cost_usd"}} (measured)
    `decisions_by_purpose`  {purpose: {action: count}} from the decision log
    """
    s = settings or get_settings()
    eq = llm_equivalent_tokens()
    rows = []
    total_llm, total_jev_replacing, total_added = 0.0, 0.0, 0.0
    for purpose in sorted(set(jev_by_purpose) | set(decisions_by_purpose)):
        base = purpose.split(":")[0]
        jev = jev_by_purpose.get(purpose, {})
        acts = decisions_by_purpose.get(purpose, {})
        jev_cost = float(jev.get("cost_usd") or 0)
        row: dict[str, Any] = {
            "purpose": purpose,
            "jev_calls": int(jev.get("calls") or 0),
            "jev_input_tokens": int(jev.get("input_tokens") or 0),
            "jev_cost_usd": round(jev_cost, 6),
            "decisions": sum(acts.values()),
            "actions": acts,
            "replaces_llm": base in REPLACES_LLM,
        }
        if base in REPLACES_LLM and base == purpose:
            # chat answers escalated to the LLM still paid for it; everything else avoided the call
            avoided = sum(v for k, v in acts.items() if k != "escalated")
            tin, tout = eq[base]
            llm = avoided * llm_cost_usd(tin, tout, s)
            row.update(
                {
                    "llm_calls_avoided": avoided,
                    "llm_equivalent_tokens": {"input": tin, "output": tout},
                    "llm_cost_avoided_usd": round(llm, 6),
                }
            )
            total_llm += llm
            total_jev_replacing += jev_cost
        elif base in REPLACES_LLM:
            total_jev_replacing += jev_cost  # e.g. "merchant_category:order_check" — part of that decision
        else:
            total_added += jev_cost
        rows.append(row)
    return {
        "rows": rows,
        "llm_cost_avoided_usd": round(total_llm, 6),
        "jev_cost_replacing_usd": round(total_jev_replacing, 6),
        "net_savings_usd": round(total_llm - total_jev_replacing, 6),
        "jev_cost_added_checks_usd": round(total_added, 6),
        "assumptions": {
            "llm_price_in_per_1m": s.llm_price_in_per_1m,
            "llm_price_out_per_1m": s.llm_price_out_per_1m,
            "jev_price_in_per_1m": s.jev_price_in_per_1m,
            "llm_equivalent_tokens": {k: {"input": v[0], "output": v[1]} for k, v in eq.items()},
            "note": "Jev spend is measured; LLM spend avoided is estimated from FinPilot's own prompts. "
            "Set LLM_PRICE_IN_PER_1M / LLM_PRICE_OUT_PER_1M to your model's prices.",
        },
    }
