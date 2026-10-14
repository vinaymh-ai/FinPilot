"""Retrieval for the chat agents: value linking over the user's ledger + knowledge lookup.

Transaction data is structured, so "RAG" here is not embedding rows into a vector store. The
accuracy problem in text-to-SQL is *schema/value linking*: knowing that "coffee" means the
Dining category or a merchant called "Third Wave Coffee", that "swigy" is "Swiggy", that "my
landlord" is "Ramesh Kumar" (category rent). This module retrieves, for one question:

* **merchants** that exist in this user's data — exact phrase, token overlap, typo-tolerant fuzzy
  match and BM25 over the merchant's raw narrations;
* **categories** implied by the wording — taxonomy labels, synonyms and the categorizer's own
  keyword vocabulary (pharmacy → health, petrol → transport);
* **knowledge snippets** — latest insights, recommendations, subscriptions, goals — ranked with BM25,
  so explanatory questions are answered from what the other agents already established.

Everything runs locally and deterministically (no embedding API), so retrieval works in zero-cost
mode and is reproducible in tests.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from finpilot.domain.lexicon import KEYWORDS
from finpilot.domain.normalize import normalize_text
from finpilot.domain.taxonomy import BY_ID, CATEGORIES, label

if TYPE_CHECKING:
    from finpilot.services.toolkit import FinanceToolkit


def _words(text: str) -> frozenset[str]:
    """Whitespace-separated vocabulary → set."""
    return frozenset(text.split())


STOPWORDS = _words(
    """
    A AN THE I ME MY MINE WE OUR YOU YOUR IT ITS IS ARE WAS WERE BE BEEN BEING AM DO DID DOES DONE HAVE HAS HAD
    WHAT WHATS WHICH WHO WHOM WHEN WHERE WHY HOW MUCH MANY ANY ALL SOME EACH EVERY PER IN ON AT TO FOR FROM BY
    WITH WITHOUT OF OFF OUT INTO OVER UNDER ABOUT ABOVE BELOW AND OR NOT NO YES SO THAN THEN THAT THIS THESE THOSE
    THERE HERE CAN COULD SHOULD WOULD WILL SHALL MAY MIGHT MUST PLEASE SHOW TELL GIVE LIST FIND GET LET SEE
    TOTAL SPEND SPENT SPENDING SPENDS PAY PAID PAYING PAYMENT PAYMENTS TRANSACTION TRANSACTIONS TXN TXNS MONEY
    AMOUNT AMOUNTS RS INR RUPEES MONTH MONTHS MONTHLY WEEK WEEKS YEAR YEARS DAY DAYS TODAY YESTERDAY LAST THIS
    NEXT PAST RECENT LATEST FIRST TIME TIMES SINCE UNTIL TILL BETWEEN DURING AVERAGE AVG MEAN COUNT NUMBER
    BIGGEST LARGEST HIGHEST SMALLEST LOWEST TOP MOST LEAST RELATED REGARDING ON MADE MAKE DONE ALSO JUST ONLY
    OFTEN USUALLY TYPICAL TYPICALLY BREAKDOWN BREAK DOWN TREND COMPARE VS VERSUS MORE LESS UP DOWN AGAIN
    DEBIT DEBITS CREDIT CREDITS RECEIVED RECEIVE SENT SEND WENT GO GOT INCOME EARN EARNED EXPENSE EXPENSES
    """
)
# Merchant-name tokens too generic to identify a payee on their own.
GENERIC_MERCHANT_TOKENS = _words(
    """
    STORE STORES SHOP MART INDIA PAY PAYMENTS BANK ONLINE SERVICES SERVICE TECHNOLOGIES PVT LTD LIMITED THE
    AND CO COM CENTRE CENTER INTERNET RETAIL HOTEL CAFE BOOKING TRANSFER MEMBERSHIP STREAMING SALARY ATM
    """
)


def tokens(text: str) -> list[str]:
    return [t for t in normalize_text(text).split() if t not in STOPWORDS and not t.isdigit() and len(t) > 1]


def _stem(token: str) -> str:
    for suffix in ("IES", "ES", "S", "ING", "ED"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("Y" if suffix == "IES" else "")
    return token


class BM25:
    """Minimal Okapi BM25 over short documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs = [[_stem(t) for t in d] for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = (sum(len(d) for d in self.docs) / len(self.docs)) if self.docs else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: list[str]) -> list[float]:
        q = [_stem(t) for t in query]
        out = []
        for d, tf in zip(self.docs, self.tf, strict=True):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += (
                    self.idf[t]
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl or 1)))
                )
            out.append(s)
        return out


@dataclass
class MerchantStats:
    name: str
    count: int = 0
    debit_total: float = 0.0
    credit_total: float = 0.0
    categories: Counter[str] = field(default_factory=Counter)
    first: str = ""
    last: str = ""
    samples: list[str] = field(default_factory=list)

    def to_dict(self, score: float | None = None, why: str | None = None) -> dict[str, Any]:
        out = {
            "merchant": self.name,
            "transactions": self.count,
            "debits": round(self.debit_total, 2),
            "credits": round(self.credit_total, 2),
            "categories": [label(c) for c, _ in self.categories.most_common(3)],
            "first_date": self.first,
            "last_date": self.last,
        }
        if score is not None:
            out["match_score"] = round(score, 2)
        if why:
            out["matched_by"] = why
        return out


# Plain-language words → category ids, beyond the taxonomy labels themselves.
CATEGORY_WORDS: dict[str, list[str]] = {
    "FOOD": ["food_delivery", "dining"],
    "EATING OUT": ["dining"],
    "EAT OUT": ["dining"],
    "CREDIT CARD": ["credit_card_payment"],
    "CARD BILL": ["credit_card_payment"],
    "CARD BILLS": ["credit_card_payment"],
    "CARD REPAYMENT": ["credit_card_payment"],
    "CARD REPAYMENTS": ["credit_card_payment"],
    "MOBILE BILL": ["utilities"],
    "PHONE BILL": ["utilities"],
    "RECHARGE": ["utilities"],
    "RECHARGES": ["utilities"],
    "FOOD DELIVERY": ["food_delivery"],
    "DINING OUT": ["dining"],
    "RESTAURANT": ["dining"],
    "DELIVERY": ["food_delivery"],
    "TAKEOUT": ["food_delivery"],
    "GROCERY": ["groceries"],
    "GROCERIES": ["groceries"],
    "VEGETABLES": ["groceries"],
    "RENT": ["rent"],
    "LANDLORD": ["rent"],
    "HOUSING": ["rent"],
    "BILL": ["utilities"],
    "BILLS": ["utilities"],
    "UTILITY": ["utilities"],
    "UTILITIES": ["utilities"],
    "ELECTRICITY": ["utilities"],
    "INTERNET": ["utilities"],
    "BROADBAND": ["utilities"],
    "TRANSPORT": ["transport"],
    "COMMUTE": ["transport"],
    "CAB": ["transport"],
    "CABS": ["transport"],
    "TAXI": ["transport"],
    "FUEL": ["transport"],
    "PETROL": ["transport"],
    "TRAVEL": ["travel"],
    "TRIP": ["travel"],
    "TRIPS": ["travel"],
    "FLIGHT": ["travel"],
    "FLIGHTS": ["travel"],
    "SHOPPING": ["shopping"],
    "CLOTHES": ["shopping"],
    "ELECTRONICS": ["shopping"],
    "SUBSCRIPTION": ["subscriptions"],
    "SUBSCRIPTIONS": ["subscriptions"],
    "STREAMING": ["subscriptions"],
    "OTT": ["subscriptions"],
    "ENTERTAINMENT": ["entertainment"],
    "MOVIE": ["entertainment"],
    "MOVIES": ["entertainment"],
    "HEALTH": ["health"],
    "MEDICAL": ["health"],
    "MEDICINE": ["health"],
    "MEDICINES": ["health"],
    "DOCTOR": ["health"],
    "PHARMACY": ["health"],
    "GYM": ["health"],
    "FITNESS": ["health"],
    "EDUCATION": ["education"],
    "COURSE": ["education"],
    "COURSES": ["education"],
    "BOOKS": ["education"],
    "SCHOOL": ["education"],
    "SALON": ["personal_care"],
    "GROOMING": ["personal_care"],
    "HAIRCUT": ["personal_care"],
    "EMI": ["loans"],
    "EMIS": ["loans"],
    "LOAN": ["loans"],
    "LOANS": ["loans"],
    "INSURANCE": ["insurance"],
    "PREMIUM": ["insurance"],
    "INVESTMENT": ["investments"],
    "INVESTMENTS": ["investments"],
    "SIP": ["investments"],
    "SIPS": ["investments"],
    "MUTUAL": ["investments"],
    "TRANSFER": ["transfers"],
    "TRANSFERS": ["transfers"],
    "FRIENDS": ["transfers"],
    "CASH": ["cash"],
    "ATM": ["cash"],
    "WITHDRAWAL": ["cash"],
    "WITHDRAWALS": ["cash"],
    "FEE": ["fees"],
    "FEES": ["fees"],
    "CHARGES": ["fees"],
    "PENALTY": ["fees"],
    "GIFT": ["gifts_donations"],
    "GIFTS": ["gifts_donations"],
    "DONATION": ["gifts_donations"],
    "DONATIONS": ["gifts_donations"],
    "CHARITY": ["gifts_donations"],
    "SALARY": ["income"],
    "INCOME": ["income"],
}


class LedgerRetriever:
    """Value linking (merchants, categories) and knowledge retrieval for one user's question."""

    def __init__(self, kit: FinanceToolkit):
        self.kit = kit
        stats: dict[str, MerchantStats] = {}
        descs: dict[str, list[str]] = defaultdict(list)
        for t in sorted(kit.txns, key=lambda x: x.date):
            m = stats.setdefault(t.merchant, MerchantStats(t.merchant))
            m.count += 1
            if t.amount < 0:
                m.debit_total += -t.amount
            else:
                m.credit_total += t.amount
            m.categories[t.category or "other"] += 1
            m.first = m.first or t.date.isoformat()
            m.last = t.date.isoformat()
            if len(m.samples) < 3 and t.description not in m.samples:
                m.samples.append(t.description)
            if len(descs[t.merchant]) < 25:
                descs[t.merchant].append(t.description)
        self.merchants = stats
        # every word that occurs in a payee name or narration (for "is this word in my data?")
        self.vocabulary: set[str] = set()
        for t in kit.txns:
            self.vocabulary |= set(normalize_text(t.merchant + " " + t.description).split())
        self._names = list(stats)
        self._name_norm = {n: normalize_text(n) for n in self._names}
        self._bm25 = BM25([tokens(n + " " + " ".join(descs[n])) for n in self._names])

    # ------------------------------------------------------------------ merchants
    def match_merchants(self, question: str, k: int = 6, min_score: float = 0.55) -> list[dict[str, Any]]:
        q_norm = normalize_text(question)
        q_tokens = [t for t in q_norm.split() if t not in STOPWORDS]
        grams = {" ".join(q_tokens[i : i + n]) for n in (1, 2, 3) for i in range(len(q_tokens) - n + 1)}
        bm = self._bm25.scores(tokens(question)) if self._names else []
        top_bm = max(bm, default=0.0) or 1.0
        scored: list[tuple[float, str, str]] = []
        for i, name in enumerate(self._names):
            norm = self._name_norm[name]
            words = [w for w in norm.split() if w not in GENERIC_MERCHANT_TOKENS] or norm.split()
            score, why = 0.0, ""
            if norm and re.search(rf"(?<![A-Z0-9]){re.escape(norm)}(?![A-Z0-9])", q_norm):
                score, why = 1.0, "exact name"
            elif words and all(re.search(rf"(?<![A-Z0-9]){re.escape(w)}(?![A-Z0-9])", q_norm) for w in words):
                score, why = 0.95, "all name words"
            else:
                hits = [
                    w for w in words if len(w) >= 4 and re.search(rf"(?<![A-Z0-9]){re.escape(w)}", q_norm)
                ]
                if hits:
                    score, why = 0.6 + 0.3 * len(hits) / len(words), "name word " + ", ".join(hits)
                best = max(
                    (SequenceMatcher(None, g, norm).ratio() for g in grams if abs(len(g) - len(norm)) <= 3),
                    default=0.0,
                )
                if best >= 0.84 and best * 0.95 > score:
                    score, why = best * 0.95, f"spelling match ({best:.2f})"
                if bm and bm[i] > 0 and 0.5 * bm[i] / top_bm > score and bm[i] >= 2.0:
                    score, why = 0.5 * bm[i] / top_bm, "narration text"
            if score >= min_score:
                scored.append((score, name, why))
        scored.sort(key=lambda x: (-x[0], -self.merchants[x[1]].count))
        return [self.merchants[n].to_dict(s, why) for s, n, why in scored[:k]]

    # ------------------------------------------------------------------ categories
    def match_categories(self, question: str) -> list[dict[str, Any]]:
        q = normalize_text(question)
        found: dict[str, tuple[float, str]] = {}

        def add(cat: str, score: float, why: str) -> None:
            if cat in BY_ID and score > found.get(cat, (0.0, ""))[0]:
                found[cat] = (score, why)

        hits: list[tuple[int, int, list[str], float, str]] = []
        for c in CATEGORIES:
            for phrase in {c.label.upper().replace("&", " "), c.id.upper().replace("_", " ")}:
                phrase = re.sub(r"\s+", " ", phrase).strip()
                for m in re.finditer(rf"(?<![A-Z]){re.escape(phrase)}(?![A-Z])", q):
                    hits.append((m.start(), m.end(), [c.id], 1.0, f"category name '{phrase.lower()}'"))
        for word, cats in CATEGORY_WORDS.items():
            for m in re.finditer(rf"(?<![A-Z]){re.escape(word)}(?![A-Z])", q):
                hits.append((m.start(), m.end(), cats, 0.9, f"word '{word.lower()}'"))
        # A longer phrase hides the words inside it: "card bill" is not a utility bill and
        # "food delivery" does not also mean dining.
        for start, end, cats, score, why in hits:
            if any(o[0] <= start and end <= o[1] and o[1] - o[0] > end - start for o in hits):
                continue
            for cat in cats:
                add(cat, score, why)
        for cat, kws in KEYWORDS.items():
            for kw, weight in kws:
                if weight >= 0.75 and re.search(rf"(?<![A-Z]){re.escape(kw)}(?![A-Z])", q):
                    add(cat, 0.7 * weight, f"keyword '{kw.lower()}'")
        return [
            {"category": c, "label": label(c), "score": round(s, 2), "matched_by": why}
            for c, (s, why) in sorted(found.items(), key=lambda kv: -kv[1][0])
        ]

    # ------------------------------------------------------------------ knowledge
    def knowledge(self, question: str, k: int = 5) -> list[dict[str, Any]]:
        from finpilot.services import repository as repo

        docs: list[dict[str, Any]] = []
        session, uid = self.kit.session, self.kit.user_id
        for i in repo.latest_insights(session, uid)[:20]:
            docs.append({"type": "insight", "title": i.title, "text": i.detail, "severity": i.severity})
        for r in repo.latest_recommendations(session, uid)[:10]:
            docs.append(
                {
                    "type": "recommendation",
                    "title": r.title,
                    "text": f"{r.message} Action: {r.action}",
                    "trigger": r.trigger,
                }
            )
        for g in self.kit.goals:
            docs.append(
                {
                    "type": "goal",
                    "title": g.name,
                    "text": f"Target {g.target}, saved {g.saved}, deadline {g.deadline.isoformat()}",
                }
            )
        if not docs:
            return []
        bm = BM25([tokens(d["title"] + " " + d["text"]) for d in docs])
        scores = bm.scores(tokens(question))
        ranked = sorted(zip(scores, range(len(docs)), strict=True), key=lambda x: -x[0])
        return [dict(docs[i], score=round(s, 2)) for s, i in ranked[:k] if s > 0]

    def context(self, question: str) -> dict[str, Any]:
        """Everything retrieved for one question (shown to the planner and the SQL analyst)."""
        return {
            "merchants": self.match_merchants(question),
            "categories": self.match_categories(question),
            "knowledge": self.knowledge(question, k=4),
        }
