"""Faithfulness guard — System One checks System Two's writing.

LLM-written text reaches the user only if it passes two independent checks:

1. **Numeric grounding (code)** — every figure in the draft must come from the facts it was given
   (`finpilot.llm.grounding`). Catches invented numbers.
2. **Claim support (Jev)** — a Choice over {supported, unsupported_claim, contradiction, generic}
   with the facts and the draft as state. Catches what numbers can't: an invented cause ("because
   of your Goa trip"), the wrong merchant, a reversed trend, or boilerplate advice.

A draft is shown only when P(supported) ≥ `JEV_GUARD_CONFIDENCE`; otherwise the deterministic
template stands. This replaces a second "LLM-as-judge" call: the check costs input tokens only and
returns a calibrated probability instead of a judge's prose. If Jev is unavailable the guard
reports `checked=False, ok=False` and the deterministic text is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from finpilot.decisions.jev import Evaluation, JevClient
from finpilot.decisions.policy import log_decisions, record
from finpilot.decisions.spec import faithfulness_questions, faithfulness_state


@dataclass
class GuardVerdict:
    checked: bool  # False → Jev not consulted (unavailable / failed); rely on the numeric check
    ok: bool  # the draft may be shown
    label: str  # supported | unsupported_claim | contradiction | generic | unchecked
    p_supported: float | None = None
    confidence: float | None = None

    def note(self) -> str:
        if not self.checked:
            return "faithfulness not checked (Jev unavailable)"
        verdict = "passed" if self.ok else "failed"
        return f"Jev faithfulness check {verdict}: {self.label.replace('_', ' ')} (P(supported)={self.p_supported:.2f})"


UNCHECKED = GuardVerdict(checked=False, ok=False, label="unchecked")


def check_faithfulness(
    jev: JevClient,
    items: list[tuple[str, Any, str]],
    *,
    agent: str,
    enabled: bool = True,
) -> list[GuardVerdict]:
    """Check drafts against their facts. `items` = [(subject, facts, draft)]; one verdict per item."""
    if not items:
        return []
    if not enabled or not jev.available():
        return [UNCHECKED for _ in items]
    jobs = [(faithfulness_state(facts, draft), faithfulness_questions()) for _, facts, draft in items]
    evals = jev.evaluate_many(jobs, agent=agent, purpose="faithfulness")
    threshold = jev.guard_confidence
    out: list[GuardVerdict] = []
    decisions = []
    for (subject, _facts, _draft), ev in zip(items, evals, strict=True):
        if not isinstance(ev, Evaluation):
            out.append(UNCHECKED)
            continue
        a = ev["support"]
        p = a.p("supported")
        ok = p >= threshold
        out.append(GuardVerdict(True, ok, a.choice, round(p, 3), round(a.confidence, 3)))
        decisions.append(
            record(
                ev,
                "support",
                agent=agent,
                purpose="faithfulness",
                subject=subject,
                action="accepted" if ok else "rejected",
            )
        )
    log_decisions(decisions)
    return out
