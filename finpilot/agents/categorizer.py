"""Categorizer agent — labels every transaction with a category, a confidence and a reason.

Cascade (cheapest, most certain first):
    1. memory   — the user's own past corrections for this merchant (learning from feedback)
    2. history  — how this merchant was categorised before (by Jev, the LLM or the user)
    3. rule     — curated merchant lexicon (≈200 Indian + global brands)
    4. keyword  — generic token evidence ("MEDICALS", "PETROL", "SALON"), amount- and sign-aware
    5. p2p      — a UPI payment to/from an individual is a person-to-person transfer; the purpose
                  is not stated, so it is labelled with review-level confidence
    6. llm      — semantic reasoning over *uncertain* merchant groups only (one decision per
                  merchant + direction), using brand knowledge, narration, amount and channel
    7. jev      — optional legacy decision model for remaining uncertainty
    8. pattern  — recurring cadence supports a subscription only when the description also
                  identifies a subscription or membership; cadence alone cannot prove purpose
    9. consistency — a merchant's uncertain rows inherit the category its confident rows agree on
The output therefore genuinely depends on each transaction's text, amount, sign and history.
Every result carries a source, a confidence and a human-readable reason (the audit trail).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from finpilot.agents.base import BaseAgent
from finpilot.decisions.jev import ChoiceAnswer, Evaluation
from finpilot.decisions.policy import DecisionRecord, log_decisions, order_debias_many, record
from finpilot.decisions.spec import merchant_questions, merchant_state
from finpilot.domain.lexicon import keyword_scores, match_merchant
from finpilot.domain.normalize import (
    detect_channel,
    is_card_repayment,
    looks_like_person,
    merchant_key,
    payment_provider,
)
from finpilot.domain.recurring import TxnLite, detect_recurring
from finpilot.domain.taxonomy import BY_ID, CATEGORIES, CATEGORY_IDS

CREDIT_CATEGORIES = {"income", "transfers", "investments"}
LLM_THRESHOLD = 0.7
P2P_CONFIDENCE = 0.6
# Results the LLM does not re-check: the user's own corrections and imported labels, and earlier
# model decisions remembered for the same merchant.
SETTLED_SOURCES = {"memory", "history", "user", "import"}


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
    jev_merchants: int = 0
    jev_accepted: int = 0
    jev_deferred: int = 0
    jev_escalated: int = 0
    jev_errors: int = 0
    jev_tokens: int = 0
    jev_cost_usd: float = 0.0
    order_checked: int = 0
    order_flipped: int = 0
    decision_model: str = "off"

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_source": dict(self.by_source),
            "llm_merchants": self.llm_merchants,
            "llm_batches": self.llm_batches,
            "llm_error": self.llm_error,
            "model": self.model,
            "decision_model": self.decision_model,
            "jev": {
                "merchants": self.jev_merchants,
                "accepted": self.jev_accepted,
                "deferred": self.jev_deferred,
                "escalated": self.jev_escalated,
                "errors": self.jev_errors,
                "input_tokens": self.jev_tokens,
                "cost_usd": round(self.jev_cost_usd, 6),
                "order_checked": self.order_checked,
                "order_flipped": self.order_flipped,
            },
        }


@dataclass
class MerchantDecision:
    """What System One decided about one merchant — persisted as a MerchantProfile."""

    merchant: str
    credit: bool
    category: str
    confidence: float
    payee_type: str | None
    service_family: str | None
    probabilities: dict[str, float]
    accepted: bool


SYSTEM_PROMPT = """You are the Categorizer agent of FinPilot, an expert in Indian retail-banking narrations
(UPI, POS/card, NEFT/IMPS/RTGS, NACH/ECS auto-debits, ATM, cheque) and in Indian and global merchant brands.
Each input group is ONE merchant seen from one direction: its cleaned name, up to 6 raw narrations with signed
amounts (negative = debit, money out; positive = credit, money in), dates, the payment channel, any payment
intermediary, and a fallible rule-based guess with its reason. Assign exactly ONE category id per group:
{taxonomy}

Reason in this order (do not output the steps):
1. Payee: identify the actual business from the narration and your brand knowledge (e.g. "MEGHANA FOODS" is a
   biryani restaurant, "BBLUNT" a salon chain, "CROSSWORD" a bookstore, "IMAGICA" an amusement park,
   "SANGEETHA MOBILES" a phone retailer, "AMUL" a dairy brand, "RAPIDO" a bike-taxi app). If the payee is
   an individual's name, it is a person.
2. Purpose of THIS payment: explicit words (RENT, EMI, SIP, SALARY, REFUND, ORDER, BILL, FEES, PREMIUM) beat
   brand defaults. UPI, Razorpay, PayU, Cashfree, CRED, Paytm and PhonePe are payment channels, not purposes.
3. Direction: a credit from a shop is a refund (use the shop's category); from an employer or client it is
   income; from an individual it is a transfer. A positive amount alone does not prove income.
4. Amount sanity: "HOTEL <name>" under 2,500 is usually a restaurant meal (dining); larger is lodging (travel).

Category rules:
- transfers: money to/from individuals (P2P) or own accounts. For a person with no stated purpose use
  transfers with confidence 0.6, because the person could be a landlord or service provider.
- subscriptions: recurring digital services and memberships only — never inferred from cadence alone.
- credit_card_payment: explicit card-bill evidence, or an ordinary payment to a bill-payment platform
  (say it is an inference). Card bills do not reveal the original purchases.
- other: only when the business and purpose genuinely cannot be identified.

Confidence calibration: 0.95 explicit in the narration or a famous single-purpose brand; 0.85 strong brand or
keyword knowledge; 0.7 plausible inference; below 0.6 is a guess — answer "other" instead of guessing.
Respond with JSON only: {{"results": [{{"id": <id>, "category": "<category id>", "confidence": <0..1>,
"merchant": "<clean merchant name>", "payee_type": "business|person|institution|own_account",
"reason": "<max 15 words: the narration words or brand knowledge that decided it>"}}]}}
Return one result for every input id."""


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
        batch_size: int = 12,
        currency: str = "INR",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.memory = {k.upper(): v for k, v in (memory or {}).items()}
        self.history = {k.upper(): v for k, v in (history or {}).items()}
        self.batch_size = batch_size
        self.currency = currency
        self.profiles: list[MerchantDecision] = []  # filled by the Jev stage, persisted by the coordinator

    # ------------------------------------------------------------------ deterministic stages
    def rule_categorize(self, item: CatItem) -> CatResult:
        merchant, norm = merchant_key(item.description)
        mk = merchant.upper()
        provider = payment_provider(item.description)
        if mk in self.memory and not provider:
            return CatResult(
                item.key,
                merchant,
                self.memory[mk],
                "memory",
                0.99,
                f"You previously categorised {merchant} as {BY_ID[self.memory[mk]].label}",
            )
        similar = next((k for k in self.memory if len(k) >= 4 and mk.startswith(k + " ")), None)
        if similar and not provider:
            cat = self.memory[similar]
            return CatResult(
                item.key,
                merchant,
                cat,
                "memory",
                0.9,
                f"Looks like {similar.title()}, which you categorised as {BY_ID[cat].label}",
            )
        if mk in self.history and not provider:
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
        if is_card_repayment(item.description):
            return CatResult(
                item.key,
                merchant,
                "credit_card_payment",
                "rule",
                0.95,
                "Explicit credit-card repayment evidence in the statement",
            )
        if provider:
            purpose = match_merchant(norm)
            if purpose and purpose.category not in {"transfers", "income", "credit_card_payment"}:
                return CatResult(
                    item.key,
                    merchant,
                    purpose.category,
                    "rule",
                    0.9,
                    f"Explicit purpose: {purpose.name} identified behind {provider}",
                )
            scores = keyword_scores(norm)
            if scores:
                category, confidence = max(scores.items(), key=lambda item: item[1])
                if confidence >= 0.85 and category not in {"transfers", "income"}:
                    return CatResult(
                        item.key,
                        merchant,
                        category,
                        "keyword",
                        confidence,
                        f"Explicit purpose: {BY_ID[category].label} in the payment note",
                    )
            return CatResult(
                item.key,
                merchant,
                "other",
                "review",
                0.3,
                f"{provider} is a payment intermediary; purchase or bill purpose needs confirmation",
            )
        rule = match_merchant(norm)
        if rule is not None and not (rule.category == "income" and not credit):
            cat = rule.category
            reason = f"Known merchant '{rule.name}'"
            if credit and cat not in CREDIT_CATEGORIES:
                reason += " (credit → refund offsets this category)"
            return CatResult(item.key, merchant, cat, "rule", 0.95, reason)

        scores = keyword_scores(norm)
        if looks_like_person(item.description) and max(scores.values(), default=0.0) < 0.5:
            # Person-to-person: the transaction kind is a transfer, but the purpose is unknown,
            # so it stays below the review threshold until the user confirms it.
            return CatResult(
                item.key,
                merchant,
                "transfers",
                "p2p",
                P2P_CONFIDENCE,
                (
                    "Money received from an individual over UPI (person-to-person)"
                    if credit
                    else "Payment to an individual over UPI (person-to-person); the purpose isn't stated "
                    "— confirm if it was rent, a service or a purchase"
                ),
            )

        tokens = set(norm.split())
        if "HOTEL" in tokens:  # amount-aware disambiguation (restaurant vs lodging)
            target = "dining" if abs(item.amount) < 2500 else "travel"
            scores[target] = 1 - (1 - scores.get(target, 0.0)) * 0.3
        if credit:
            scores = {c: s for c, s in scores.items() if c in CREDIT_CATEGORIES or c == "fees"}
            if not scores:
                return CatResult(
                    item.key,
                    merchant,
                    "other",
                    "review",
                    0.35,
                    "Credit could be income, refund or transfer; needs review",
                )
        else:
            scores.pop("income", None)
        if scores:
            cat, score = max(scores.items(), key=lambda kv: kv[1])
            matched = [t for t in tokens if any(t in kw for kw, _ in _kw_for(cat))]
            reason = f"Keyword evidence ({', '.join(sorted(matched)[:3]) or 'description'}) suggests {BY_ID[cat].label}"
            return CatResult(item.key, merchant, cat, "keyword", round(min(0.9, score), 2), reason)
        return CatResult(item.key, merchant, "other", "fallback", 0.25, "No rule or keyword matched")

    # ------------------------------------------------------------------ Jev stage (System One)
    @staticmethod
    def _group_key(item: CatItem, result: CatResult) -> tuple[str, bool]:
        # Amount-dependent labels must not spread from a restaurant meal to a hotel stay.
        key = result.merchant
        if payment_provider(item.description):
            # A platform can process rent, purchases and card bills. Never copy one purpose
            # across its other payments, even when their payee and direction match.
            key += "|" + item.description
        if "HOTEL" in result.merchant.upper():
            key += "|lodging" if abs(item.amount) >= 2500 else "|meal"
        return key, item.amount > 0

    def _uncertain_groups(
        self, items: list[CatItem], results: dict[int, CatResult], only: set[tuple[str, bool]] | None = None
    ) -> list[tuple[tuple[str, bool], list[CatItem]]]:
        """Still-uncertain transactions grouped per unique merchant + direction (one decision each)."""
        by_merchant: dict[tuple[str, bool], list[CatItem]] = defaultdict(list)
        for it in items:
            r = results[it.key]
            if r.confidence < LLM_THRESHOLD:
                key = self._group_key(it, r)
                if only is None or key in only:
                    by_merchant[key].append(it)
        return list(by_merchant.items())

    def _jev_refine(
        self, items: list[CatItem], results: dict[int, CatResult], stats: CategorizerStats
    ) -> set[tuple[str, bool]]:
        """Ask Jev about every uncertain merchant. Returns the merchants it deferred on."""
        groups = self._uncertain_groups(items, results)
        if not groups:
            return set()
        jobs = []
        for (_, credit), its in groups:
            merchant = results[its[0].key].merchant
            amounts = sorted(abs(i.amount) for i in its)
            jobs.append(
                (
                    merchant_state(
                        merchant,
                        its[0].description,
                        credit=credit,
                        typical_amount=amounts[len(amounts) // 2],
                        count=len(its),
                        currency=self.currency,
                    ),
                    merchant_questions(credit=credit, typical_amount=amounts[len(amounts) // 2]),
                )
            )
        stats.jev_merchants = len(groups)
        stats.decision_model = self.jev.describe()
        evals = self.jev.evaluate_many(jobs, agent=self.name, purpose="merchant_category")
        evals, ostats = order_debias_many(
            self.jev, jobs, evals, agent=self.name, purpose="merchant_category", names={"category"}
        )
        stats.order_checked, stats.order_flipped = ostats["checked"], ostats["flipped"]
        threshold = self.jev.accept_confidence
        deferred: set[tuple[str, bool]] = set()
        decisions: list[DecisionRecord] = []
        for (group_key, its), ev in zip(groups, evals, strict=True):
            merchant, credit = results[its[0].key].merchant, group_key[1]
            if not isinstance(ev, Evaluation):
                stats.jev_errors += 1
                deferred.add(group_key)
                continue
            stats.jev_tokens += ev.input_tokens
            stats.jev_cost_usd += ev.cost_usd
            cat: ChoiceAnswer = ev["category"]
            payee = ev.answers.get("payee_type")
            family = ev.answers.get("service_family")
            category, conf, why = cat.choice, cat.confidence, None
            ok = conf >= threshold and category != "other"
            if category == "transfers" and not re.search(
                r"SELF TRANSFER|OWN ACCOUNT|FUND TRANSFER|TRANSFER TO|\bP2P\b", its[0].description.upper()
            ):
                ok = False
            flipped = "category" in ev.meta.get("order_flipped", [])
            if ok:
                stats.jev_accepted += 1
                if why is None:
                    ru = cat.runner_up
                    why = f"Jev: {BY_ID[category].label} (p={cat.p(category):.2f}" + (
                        f"; next {BY_ID[ru[0]].label} {ru[1]:.2f})" if ru and ru[0] in BY_ID else ")"
                    )
                    if ev.meta.get("order_checked"):
                        why += " — consistent when options were reordered" if not flipped else ""
                for it in its:
                    prev = results[it.key]
                    if conf >= prev.confidence or prev.source == "fallback":
                        results[it.key] = CatResult(
                            it.key, prev.merchant, category, "jev", round(min(conf, 0.97), 2), why
                        )
            else:
                stats.jev_deferred += 1
                deferred.add(group_key)
            self.profiles.append(
                MerchantDecision(
                    merchant=merchant,
                    credit=credit,
                    category=category,
                    confidence=round(conf, 4),
                    payee_type=payee.choice if payee is not None and payee.confidence >= threshold else None,
                    service_family=(
                        family.choice if family is not None and family.confidence >= threshold else None
                    ),
                    probabilities={k: round(v, 4) for k, v in cat.probabilities.items() if v >= 0.01},
                    accepted=ok,
                )
            )
            decisions.append(
                record(
                    ev,
                    "category",
                    agent=self.name,
                    purpose="merchant_category",
                    subject=f"{merchant} ({'credit' if credit else 'debit'}, {len(its)} txns)",
                    action=("accepted_after_llm" if ok else "deferred_after_llm")
                    if self.llm_available
                    else ("accepted" if ok else "deferred"),
                )
            )
            if family is not None and family.choice not in ("none", "utility_bill"):
                decisions.append(
                    record(
                        ev,
                        "service_family",
                        agent=self.name,
                        purpose="service_family",
                        subject=merchant,
                        action="accepted" if family.confidence >= threshold else "deferred",
                    )
                )
        log_decisions(decisions)
        return deferred

    # ------------------------------------------------------------------ LLM stage
    @staticmethod
    def _settled(result: CatResult) -> bool:
        """User decisions and earlier model decisions for the same merchant are not re-asked."""
        return result.source in SETTLED_SOURCES

    def _llm_refine(
        self,
        items: list[CatItem],
        results: dict[int, CatResult],
        stats: CategorizerStats,
        only: set[tuple[str, bool]] | None = None,
    ) -> None:
        """One reasoning decision per uncertain merchant + direction (pay per merchant, not per row)."""
        groups_by_key: dict[tuple[str, bool], list[CatItem]] = defaultdict(list)
        for item in items:
            result = results[item.key]
            if self._settled(result):
                continue
            key = self._group_key(item, result)
            if only is None or key in only:
                groups_by_key[key].append(item)
        groups = list(groups_by_key.items())
        if not groups or not self.llm_available:
            return
        stats.llm_merchants += len(groups)
        for start in range(0, len(groups), self.batch_size):
            batch = groups[start : start + self.batch_size]
            payload = {
                "currency": self.currency,
                "merchants": [
                    {
                        "id": idx,
                        "merchant": results[its[0].key].merchant,
                        "payment_provider": payment_provider(its[0].description),
                        "channel": detect_channel(its[0].description),
                        "transactions": [
                            {"description": i.description, "amount": i.amount, "date": i.date}
                            for i in its[:6]
                        ],
                        "direction": "credit" if credit else "debit",
                        "typical_amount": round(sorted(abs(i.amount) for i in its)[len(its) // 2], 2),
                        "count": len(its),
                        "rule_guess": results[its[0].key].category,
                        "rule_confidence": results[its[0].key].confidence,
                        "rule_reason": results[its[0].key].reason,
                    }
                    for idx, ((_m, credit), its) in enumerate(batch, start=start)
                ],
            }
            data, label, err = self.safe_ask_json(payload, max_tokens=2500, temperature=0)
            stats.llm_batches += 1
            if data is None:
                stats.llm_error = err
                continue
            stats.model = label or stats.model
            rows = data.get("results", data) if isinstance(data, dict) else data
            for row in rows if isinstance(rows, list) else []:
                try:
                    if type(row.get("id")) is not int:
                        continue
                    idx = row["id"]
                    cat = str(row.get("category", "")).strip().lower()
                    conf = float(row.get("confidence", 0.6))
                except (TypeError, ValueError, AttributeError):
                    continue
                if (
                    cat not in CATEGORY_IDS
                    or not (start <= idx < start + len(batch))
                    or not math.isfinite(conf)
                    or not 0 <= conf <= 1
                ):
                    continue
                (_m, _credit), its = groups[idx]
                reason = str(row.get("reason") or row.get("evidence") or "LLM classification")[:200]
                if conf < LLM_THRESHOLD or cat == "other":
                    reason = f"Uncertain {cat}: {reason}"
                    cat, conf = "other", min(conf, LLM_THRESHOLD - 0.01)
                for it in its:
                    prev = results[it.key]
                    if is_card_repayment(it.description) and cat != "credit_card_payment":
                        continue  # explicit repayment evidence outranks a model's unsupported inference
                    if cat == "other" and prev.category != "other":
                        # An unsure model must not erase specific evidence (a keyword, a P2P payee).
                        results[it.key] = CatResult(
                            it.key,
                            prev.merchant,
                            prev.category,
                            prev.source,
                            prev.confidence,
                            f"{prev.reason} (model review could not confirm)",
                        )
                        continue
                    if cat == prev.category:
                        # Independent agreement between rules and the model raises confidence.
                        results[it.key] = CatResult(
                            it.key,
                            prev.merchant,
                            cat,
                            "llm",
                            round(min(max(conf, prev.confidence) + 0.03, 0.97), 2),
                            f"LLM agrees with rules: {reason}",
                        )
                    elif prev.source == "rule" and prev.confidence >= 0.9 and conf < 0.9:
                        # A curated lexicon / explicit-evidence rule yields only to a confident model.
                        results[it.key] = CatResult(
                            it.key,
                            prev.merchant,
                            prev.category,
                            prev.source,
                            prev.confidence,
                            f"{prev.reason} (model suggested {BY_ID[cat].label} at {conf:.2f}; rule kept)",
                        )
                    elif conf >= LLM_THRESHOLD or prev.confidence < LLM_THRESHOLD:
                        results[it.key] = CatResult(
                            it.key, prev.merchant, cat, "llm", round(min(conf, 0.95), 2), f"LLM: {reason}"
                        )

    # ------------------------------------------------------------------ consistency stage
    @classmethod
    def consistency_refine(cls, results: dict[int, CatResult], items: list[CatItem]) -> int:
        """Uncertain rows of a merchant inherit the category its confident rows agree on.

        A merchant often appears in several narration styles (POS one day, UPI the next); when every
        confident label for it, in the same direction, says the same thing, the stragglers follow.
        Payment intermediaries are excluded (each payment can have a different purpose).
        """
        by_merchant: dict[tuple[str, bool], list[CatItem]] = defaultdict(list)
        for it in items:
            if payment_provider(it.description):
                continue
            by_merchant[cls._group_key(it, results[it.key])].append(it)
        changed = 0
        for (_merchant, _credit), its in by_merchant.items():
            confident = [
                results[i.key]
                for i in its
                if results[i.key].confidence >= 0.8 and results[i.key].category != "other"
            ]
            labels = {r.category for r in confident}
            if len(labels) != 1:
                continue
            (category,) = labels
            for it in its:
                r = results[it.key]
                if r.category == category or r.confidence >= 0.8 or r.source in SETTLED_SOURCES:
                    continue
                results[it.key] = CatResult(
                    it.key,
                    r.merchant,
                    category,
                    "consistency",
                    0.8,
                    f"Same merchant as {len(confident)} confidently labelled transaction(s) "
                    f"({BY_ID[category].label})",
                )
                changed += 1
        return changed

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
            item_by_key = {i.key: i for i in items}
            for k in series.txn_ids:
                r = results[k]
                if not re.search(r"SUBSCRIPTION|MEMBERSHIP|MONTHLY PLAN", item_by_key[k].description.upper()):
                    continue
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
        stats = CategorizerStats(total=len(items), model="rules")
        results = {it.key: self.rule_categorize(it) for it in items}
        self.profiles = []
        # Generative reasoning sees all uncertain narrations, including intermediaries.
        # Jev handles any remaining uncertainty; an optional cascade can retry those
        # decisions, but cannot switch off the initial reasoning stage.
        self._llm_refine(items, results, stats)
        unresolved = [
            it
            for it in items
            if results[it.key].source == "llm"
            and results[it.key].category == "other"
            and payment_provider(it.description)
        ]
        if unresolved and self.llm_available:
            original = self.system_prompt
            self.system_prompt = (
                original
                + "\nSecond reasoning level: audit your uncertain intermediary classifications. Evaluate the payee's business separately from the payment purpose. A company name is not a paid product; reject invented membership fees. Consider ordinary bill-payment use when no contrary purpose is stated, label it as inference with calibrated uncertainty. Retain other if genuine ambiguity remains. Return the same JSON schema."
            )
            try:
                self._llm_refine(unresolved, results, stats)
            finally:
                self.system_prompt = original
        if self.jev_available:
            deferred = self._jev_refine(items, results, stats)
            # Cascading low-confidence answers to an LLM is opt-in: in independent tests the fallback
            # model fixed fewer errors than it introduced. Enable only after `run_eval --jev` shows a gain.
            if deferred and self.jev.escalate_to_llm and self.llm_available:
                stats.jev_escalated = len(deferred)
                self._llm_refine(items, results, stats, only=deferred)
        if use_patterns:
            self.pattern_refine(results, items)
        self.consistency_refine(results, items)
        for r in results.values():
            stats.by_source[r.source] += 1
        return [results[it.key] for it in items], stats


def _kw_for(cat: str) -> list[tuple[str, float]]:
    from finpilot.domain.lexicon import KEYWORDS

    return KEYWORDS.get(cat, [])
