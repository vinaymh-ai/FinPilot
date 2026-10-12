"""Categorizer agent — labels every transaction with a category, a confidence and a reason.

Cascade (cheapest, most certain first):
    1. memory   — the user's own past corrections for this merchant (learning from feedback)
    2. history  — how this merchant was categorised before (by the LLM or the user)
    3. rule     — curated merchant lexicon (≈200 Indian + global brands)
    4. keyword  — generic token evidence ("MEDICALS", "PETROL", "SALON"), amount- and sign-aware
    5. llm      — batched, per *unique merchant*, only for what is still uncertain
    6. pattern  — post-pass: an unknown merchant charging the same amount every month is a
                  subscription, whatever its name says
The output therefore genuinely depends on each transaction's text, amount, sign and history.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from finpilot.agents.base import BaseAgent
from finpilot.domain.lexicon import keyword_scores, match_merchant
from finpilot.domain.normalize import looks_like_person, merchant_key
from finpilot.domain.recurring import TxnLite, detect_recurring
from finpilot.domain.taxonomy import BY_ID, CATEGORIES, CATEGORY_IDS

CREDIT_CATEGORIES = {"income", "transfers", "investments"}
LLM_THRESHOLD = 0.7


@dataclass
class CatItem:
    key: int  # transaction id (or row index)
    description: str
    amount: float
    date: dt.date | None = None


@dataclass
class CatResult:
    key: int
    merchant: str
    category: str
    source: str
    confidence: float
    reason: str


@dataclass
class CategorizerStats:
    total: int = 0
    by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    llm_merchants: int = 0
    llm_batches: int = 0
    llm_error: str | None = None
    model: str = "rules"

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_source": dict(self.by_source),
            "llm_merchants": self.llm_merchants,
            "llm_batches": self.llm_batches,
            "llm_error": self.llm_error,
            "model": self.model,
        }


SYSTEM_PROMPT = """You are the Categorizer agent of FinPilot, a personal-finance co-pilot (default market: India, INR).
You receive bank-statement merchants that our rule engine could not classify confidently.
Assign each one exactly ONE category id from this taxonomy:
{taxonomy}

Guidelines:
- Use world knowledge of brands and Indian retail (e.g. "TRUFFLES" is a Bengaluru restaurant chain, "GREEN TRENDS" is a salon chain, "CHUMBAK" is a lifestyle store).
- In India "HOTEL <name>" with a small amount is usually a restaurant (dining); with a large amount it is lodging (travel).
- A person's name paid via UPI is a person-to-person transfer ("transfers") unless the note says otherwise (e.g. RENT).
- Credits (positive amounts) are usually income, refunds or transfers.
- If you genuinely cannot tell, answer "other" with a low confidence.
Respond with JSON only: {{"results": [{{"id": <id>, "category": "<category id>", "confidence": <0..1>, "merchant": "<clean merchant name>", "reason": "<max 12 words>"}}]}}"""


def _taxonomy_text() -> str:
    return "\n".join(f"- {c.id}: {c.label} — {c.description}" for c in CATEGORIES)


class CategorizerAgent(BaseAgent):
    name = "categorizer"
    role = "Labels each transaction (rent, food, subscriptions, impulse…) with confidence and reason"
    system_prompt = SYSTEM_PROMPT.format(taxonomy=_taxonomy_text())

    def __init__(
        self,
        *args: Any,
        memory: dict[str, str] | None = None,
        history: dict[str, str] | None = None,
        batch_size: int = 40,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.memory = {k.upper(): v for k, v in (memory or {}).items()}
        self.history = {k.upper(): v for k, v in (history or {}).items()}
        self.batch_size = batch_size

    # ------------------------------------------------------------------ deterministic stages
    def rule_categorize(self, item: CatItem) -> CatResult:
        merchant, norm = merchant_key(item.description)
        mk = merchant.upper()
        if mk in self.memory:
            return CatResult(
                item.key,
                merchant,
                self.memory[mk],
                "memory",
                0.99,
                f"You previously categorised {merchant} as {BY_ID[self.memory[mk]].label}",
            )
        similar = next((k for k in self.memory if len(k) >= 4 and mk.startswith(k + " ")), None)
        if similar:
            cat = self.memory[similar]
            return CatResult(
                item.key,
                merchant,
                cat,
                "memory",
                0.9,
                f"Looks like {similar.title()}, which you categorised as {BY_ID[cat].label}",
            )
        if mk in self.history:
            cat = self.history[mk]
            return CatResult(
                item.key,
                merchant,
                cat,
                "history",
                0.9,
                f"Same merchant was categorised as {BY_ID[cat].label} before",
            )

        credit = item.amount > 0
        rule = match_merchant(norm)
        if rule is not None and not (rule.category == "income" and not credit):
            cat = rule.category
            reason = f"Known merchant '{rule.name}'"
            if credit and cat not in CREDIT_CATEGORIES:
                reason += " (credit → refund offsets this category)"
            return CatResult(item.key, merchant, cat, "rule", 0.95, reason)

        scores = keyword_scores(norm)
        if looks_like_person(item.description) and max(scores.values(), default=0.0) < 0.5:
            return CatResult(
                item.key,
                merchant,
                "transfers",
                "keyword",
                0.72,
                "UPI payment to an individual (P2P transfer)",
            )

        tokens = set(norm.split())
        if "HOTEL" in tokens:  # amount-aware disambiguation (restaurant vs lodging)
            target = "dining" if abs(item.amount) < 2500 else "travel"
            scores[target] = 1 - (1 - scores.get(target, 0.0)) * 0.3
        if credit:
            scores = {c: s for c, s in scores.items() if c in CREDIT_CATEGORIES or c == "fees"}
            if not scores:
                return CatResult(
                    item.key, merchant, "income", "keyword", 0.55, "Unrecognised credit treated as income"
                )
        else:
            scores.pop("income", None)
        if scores:
            cat, score = max(scores.items(), key=lambda kv: kv[1])
            matched = [t for t in tokens if any(t in kw for kw, _ in _kw_for(cat))]
            reason = f"Keyword evidence ({', '.join(sorted(matched)[:3]) or 'description'}) suggests {BY_ID[cat].label}"
            return CatResult(item.key, merchant, cat, "keyword", round(min(0.9, score), 2), reason)
        return CatResult(item.key, merchant, "other", "fallback", 0.25, "No rule or keyword matched")

    # ------------------------------------------------------------------ LLM stage
    def _llm_refine(
        self, items: list[CatItem], results: dict[int, CatResult], stats: CategorizerStats
    ) -> None:
        uncertain = [results[i.key] for i in items if results[i.key].confidence < LLM_THRESHOLD]
        if not uncertain or not self.llm_available:
            return
        # one query per unique merchant (+direction): 300 Swiggy-like rows cost a single line
        by_merchant: dict[tuple[str, bool], list[CatItem]] = defaultdict(list)
        item_by_key = {i.key: i for i in items}
        for r in uncertain:
            it = item_by_key[r.key]
            by_merchant[(r.merchant, it.amount > 0)].append(it)
        groups = list(by_merchant.items())
        stats.llm_merchants = len(groups)
        for start in range(0, len(groups), self.batch_size):
            batch = groups[start : start + self.batch_size]
            payload = {
                "merchants": [
                    {
                        "id": idx,
                        "description": its[0].description[:120],
                        "direction": "credit" if credit else "debit",
                        "typical_amount": round(sorted(abs(i.amount) for i in its)[len(its) // 2], 2),
                        "count": len(its),
                        "rule_guess": results[its[0].key].category,
                    }
                    for idx, ((_m, credit), its) in enumerate(batch, start=start)
                ]
            }
            data, label, err = self.safe_ask_json(payload, max_tokens=2500)
            stats.llm_batches += 1
            if data is None:
                stats.llm_error = err
                continue
            stats.model = label or stats.model
            rows = data.get("results", data) if isinstance(data, dict) else data
            for row in rows if isinstance(rows, list) else []:
                try:
                    idx = int(row.get("id"))
                    cat = str(row.get("category", "")).strip().lower()
                    conf = float(row.get("confidence", 0.6))
                except (TypeError, ValueError, AttributeError):
                    continue
                if cat not in CATEGORY_IDS or not (start <= idx < start + len(batch)):
                    continue
                (_m, _credit), its = groups[idx]
                reason = str(row.get("reason") or "LLM classification")[:200]
                for it in its:
                    prev = results[it.key]
                    if conf >= prev.confidence or prev.source == "fallback":
                        results[it.key] = CatResult(
                            it.key, prev.merchant, cat, "llm", round(min(conf, 0.95), 2), f"LLM: {reason}"
                        )

    # ------------------------------------------------------------------ pattern stage
    @staticmethod
    def pattern_refine(results: dict[int, CatResult], items: list[CatItem]) -> int:
        """Unknown merchants that bill a fixed amount monthly are subscriptions."""
        lite = [
            TxnLite(i.key, i.date, i.amount, results[i.key].merchant, results[i.key].category)
            for i in items
            if i.date is not None
        ]
        changed = 0
        for series in detect_recurring(lite):
            if series.cadence not in ("monthly", "yearly") or series.typical_amount >= 0:
                continue
            members = [results[k] for k in series.txn_ids if k in results]
            if not members or any(m.source in ("memory", "rule", "history") for m in members):
                continue
            if abs(series.typical_amount) > 5000 or all(m.category == "subscriptions" for m in members):
                continue
            if all(m.confidence >= 0.8 for m in members):
                continue
            for k in series.txn_ids:
                r = results[k]
                results[k] = CatResult(
                    k,
                    r.merchant,
                    "subscriptions",
                    "pattern",
                    0.75,
                    f"Fixed {series.cadence} charge of {abs(series.typical_amount):.0f} "
                    f"({series.occurrences}×) — behaves like a subscription",
                )
                changed += 1
        return changed

    # ------------------------------------------------------------------ public API
    def categorize(
        self, items: list[CatItem], *, use_patterns: bool = True
    ) -> tuple[list[CatResult], CategorizerStats]:
        stats = CategorizerStats(total=len(items), model=self.model_label)
        results = {it.key: self.rule_categorize(it) for it in items}
        self._llm_refine(items, results, stats)
        if use_patterns:
            self.pattern_refine(results, items)
        for r in results.values():
            stats.by_source[r.source] += 1
        return [results[it.key] for it in items], stats


def _kw_for(cat: str) -> list[tuple[str, float]]:
    from finpilot.domain.lexicon import KEYWORDS

    return KEYWORDS.get(cat, [])
