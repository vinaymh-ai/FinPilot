"""Auditor agent — independently verifies the decision trail before advice reaches the user.

The advisor explains *why* it recommends something; the auditor checks that explanation against
what the other agents actually produced in the same run:

* the trigger names a real producer (``analyzer:<finding kind>`` must match a finding of that kind
  in this run; ``forecaster:<signal>`` needs a forecast),
* the evidence is non-empty and every evidence transaction id belongs to this user's ledger,
* the reasoning trail has at least two steps and the impact is a finite, non-negative number,
* every figure in the user-facing title and message is grounded in the recommendation's own
  evidence (numeric grounding, tolerant of the advisor's ×12 / ÷12 conversions).

It also reports categorization health (share of outflows with an uncertain category), because
every downstream number depends on it. The audit verdict and its reasons are appended to each
recommendation's reasoning, so the "Why?" drawer shows that the claim was checked.
"""

from __future__ import annotations

import math
from typing import Any

from finpilot.agents.base import BaseAgent
from finpilot.llm.grounding import check_grounding


class AuditorAgent(BaseAgent):
    name = "auditor"
    role = (
        "Verifies each recommendation's trigger, evidence, figures and reasoning against this run's outputs"
    )

    def audit(
        self,
        recommendations: list[dict[str, Any]],
        *,
        finding_kinds: set[str],
        has_forecast: bool,
        ledger_ids: set[int],
        review_outflow_share: float | None = None,
    ) -> dict[str, Any]:
        results = []
        for rec in recommendations:
            issues: list[str] = []
            checks: list[str] = []
            trigger = str(rec.get("trigger") or "")
            producer, _, signal = trigger.partition(":")
            if producer == "analyzer":
                if signal in finding_kinds or signal.replace("anomaly_", "") in {
                    k.replace("anomaly_", "") for k in finding_kinds
                }:
                    checks.append(f"trigger matches analyzer finding '{signal}'")
                else:
                    issues.append(f"trigger '{trigger}' has no matching analyzer finding in this run")
            elif producer == "forecaster":
                if has_forecast:
                    checks.append("trigger matches this run's forecast")
                else:
                    issues.append("forecast trigger without a forecast")
            else:
                issues.append(f"unknown trigger '{trigger}'")
            evidence = rec.get("evidence") or {}
            if not evidence:
                issues.append("no evidence recorded")
            ids = [i for i in evidence.get("transaction_ids") or [] if isinstance(i, int)]
            if ids:
                unknown = [i for i in ids if i not in ledger_ids]
                if unknown:
                    issues.append(f"{len(unknown)} evidence transaction id(s) are not in this ledger")
                else:
                    checks.append(f"{len(ids)} evidence transaction(s) verified in the ledger")
            if len(rec.get("reasoning") or []) < 2:
                issues.append("reasoning trail has fewer than two steps")
            impact = rec.get("monthly_impact", 0)
            if not isinstance(impact, (int, float)) or not math.isfinite(impact) or impact < 0:
                issues.append("monthly impact is not a finite non-negative number")
            facts = [evidence, rec.get("monthly_impact"), rec.get("one_time_impact"), rec.get("confidence")]
            ok_numbers, bad = check_grounding(f"{rec.get('title', '')}\n{rec.get('message', '')}", *facts)
            if ok_numbers:
                checks.append("all figures grounded in the evidence")
            else:
                issues.append(f"figures not found in the evidence: {bad[:4]}")
            results.append(
                {"title": rec.get("title"), "passed": not issues, "checks": checks, "issues": issues}
            )
        passed = sum(r["passed"] for r in results)
        out: dict[str, Any] = {
            "checked": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": results,
        }
        if review_outflow_share is not None:
            out["categorization_review_share_pct"] = round(review_outflow_share, 1)
            out["categorization_health"] = (
                "good"
                if review_outflow_share < 2
                else "fair"
                if review_outflow_share < 8
                else "needs attention"
            )
        return out
