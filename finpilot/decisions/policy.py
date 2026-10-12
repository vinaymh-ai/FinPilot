"""Confidence policy, order-bias control and the decision audit log.

A Jev answer is a distribution, not a verdict — the code decides what to do with it:

    confidence ≥ threshold  → act on it            (action = "accepted")
    below threshold         → keep the safer path  (action = "deferred" / "escalated")

Two safeguards from our evaluation and TypeSafe's jev-1.13 notes:
* **Option-order bias.** Mid-confidence Choice answers are asked again with the options
  reversed and the two distributions averaged (permutation averaging). If the winner flips,
  the merged confidence collapses and the answer is deferred.
* **Thresholds are data.** Defaults are conservative; `evaluation/run_eval.py --jev` reports
  calibration and picks thresholds on a validation split, reported on a separate test split.

Every decision lands in the `decisions` table with its probabilities and the action taken —
the "why did the system do that?" trail for closed-set judgments.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from finpilot.decisions.jev import (
    Choice,
    ChoiceAnswer,
    Evaluation,
    JevClient,
    JevError,
    Question,
    confidence_from,
)
from finpilot.llm.client import current_run_id

log = logging.getLogger(__name__)

ORDER_CHECK_CEILING = 0.9  # above this Jev was well calibrated in independent tests; below, re-check


@dataclass
class DecisionRecord:
    agent: str
    purpose: str
    subject: str
    question: str
    answer: str
    confidence: float | None
    probabilities: dict[str, float] | None
    action: str  # accepted | deferred | escalated | rejected | overridden
    model: str
    latency_ms: int = 0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def top_probabilities(probs: dict[str, float], k: int = 5) -> dict[str, float]:
    return {key: round(v, 4) for key, v in sorted(probs.items(), key=lambda kv: -kv[1])[:k]}


def record(
    ev: Evaluation, name: str, *, agent: str, purpose: str, subject: str, action: str
) -> DecisionRecord:
    a = ev.answers[name]
    return DecisionRecord(
        agent=agent,
        purpose=purpose,
        subject=str(subject)[:255],
        question=name,
        answer=str(a.value)[:128],
        confidence=round(float(a.confidence), 4),
        probabilities=top_probabilities(a.probabilities),
        action=action,
        model=ev.label[:64],
        latency_ms=ev.latency_ms,
        cached=ev.cached,
    )


def log_decisions(records: Iterable[DecisionRecord]) -> int:
    rows = list(records)
    if not rows:
        return 0
    from finpilot.db import session_scope
    from finpilot.models import Decision

    try:
        with session_scope() as s:
            run_id = current_run_id()
            s.add_all(Decision(run_id=run_id, **r.to_dict()) for r in rows)
    except Exception:  # noqa: BLE001 - the audit log must never break the decision itself
        log.warning("failed to write %d decision records", len(rows), exc_info=True)
        return 0
    return len(rows)


def accepted(answer: Any, threshold: float) -> bool:
    return answer is not None and float(answer.confidence) >= threshold


# ====================================================================== order bias


def _merge(a: ChoiceAnswer, b: ChoiceAnswer, options: Sequence[str]) -> ChoiceAnswer:
    probs = {k: (a.p(k) + b.p(k)) / 2 for k in options}
    choice = max(probs, key=lambda k: probs[k])
    return ChoiceAnswer(choice, {k: round(v, 6) for k, v in probs.items()}, round(confidence_from(probs), 6))


def order_debias_many(
    client: JevClient,
    jobs: Sequence[tuple[Any, dict[str, Question]]],
    evals: Sequence[Evaluation | JevError],
    *,
    agent: str,
    purpose: str,
    names: set[str] | None = None,
    low: float | None = None,
    high: float = ORDER_CHECK_CEILING,
) -> tuple[list[Evaluation | JevError], dict[str, int]]:
    """Re-ask mid-confidence Choice answers with reversed options; average the two orders.

    Only answers in [low, high) are re-asked — confident answers are left alone, so the extra
    spend is bounded by how uncertain the batch actually is. Returns the merged evaluations and
    stats: {"checked": n, "flipped": m}.
    """
    stats = {"checked": 0, "flipped": 0}
    if not client.order_check:
        return list(evals), stats
    low = client.accept_confidence if low is None else low
    redo: list[tuple[int, dict[str, Choice]]] = []
    for i, ((_state, qs), ev) in enumerate(zip(jobs, evals, strict=True)):
        if not isinstance(ev, Evaluation):
            continue
        targets = {
            n: q
            for n, q in qs.items()
            if isinstance(q, Choice)
            and (names is None or n in names)
            and low <= ev.answers[n].confidence < high
        }
        if targets:
            redo.append((i, targets))
    if not redo:
        return list(evals), stats
    reversed_jobs = [(jobs[i][0], {n: q.reversed() for n, q in t.items()}) for i, t in redo]
    second = client.evaluate_many(reversed_jobs, agent=agent, purpose=f"{purpose}:order_check")
    out = list(evals)
    for (i, targets), rev in zip(redo, second, strict=True):
        ev = out[i]
        if not isinstance(rev, Evaluation) or not isinstance(ev, Evaluation):
            continue
        merged = dict(ev.answers)
        flips: list[str] = []
        for n, q in targets.items():
            a, b = ev.answers[n], rev.answers[n]
            assert isinstance(a, ChoiceAnswer) and isinstance(b, ChoiceAnswer)
            stats["checked"] += 1
            if a.choice != b.choice:
                stats["flipped"] += 1
                flips.append(n)
            merged[n] = _merge(a, b, list(q.criteria))
        out[i] = replace(
            ev,
            answers=merged,
            input_tokens=ev.input_tokens + rev.input_tokens,
            cost_usd=ev.cost_usd + rev.cost_usd,
            latency_ms=ev.latency_ms + rev.latency_ms,
            meta={**ev.meta, "order_checked": sorted(targets), "order_flipped": flips},
        )
    return out, stats


def order_debias(
    client: JevClient,
    state: Any,
    questions: dict[str, Question],
    ev: Evaluation,
    *,
    agent: str,
    purpose: str,
    names: set[str] | None = None,
) -> Evaluation:
    out, _ = order_debias_many(client, [(state, questions)], [ev], agent=agent, purpose=purpose, names=names)
    merged = out[0]
    return merged if isinstance(merged, Evaluation) else ev
