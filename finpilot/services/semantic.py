"""Semantic retrieval over one user's payees: hybrid sparse vectors + optional dense embeddings.

Why not only BM25? Questions use words the narrations never contain ("coffee" for Starbucks,
"medicines" for Apollo, "cab" for Rapido) and spell names the way people say them ("hanumantha"
for HANUMANTHAIAH). Each payee is turned into a document of

* **name and narration words** (TF-IDF weighted);
* **character n-grams** of the name — typo- and transliteration-tolerant matching;
* **meaning** — the payee's category (label + description + vocabulary) and the merchant
  enrichment descriptors ("Starbucks: coffee cafe chain").

Queries get the same features, plus the categories their words imply. Cosine similarity ranks the
payees. When an embedding model is configured (`EMBEDDING_MODEL`), dense vectors from the API are
fused with the sparse ranking by Reciprocal Rank Fusion; without one, everything stays local,
deterministic and free.

`name_variants` groups spellings of the same payee ("HANUMANTHAIAH" / "HANUMANTHIAH"), using a
consonant skeleton tuned for Indian names plus edit similarity.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from finpilot.domain.enrichment import describe
from finpilot.domain.normalize import normalize_text
from finpilot.domain.taxonomy import BY_ID

if TYPE_CHECKING:
    from finpilot.llm.client import LLMRouter

log = logging.getLogger(__name__)

_DIGRAPHS = (("TH", "T"), ("DH", "D"), ("BH", "B"), ("KH", "K"), ("GH", "G"), ("PH", "F"), ("SH", "S"))
_DIGRAPHS += (("CH", "C"), ("W", "V"), ("Z", "J"), ("Q", "K"), ("CK", "K"), ("X", "KS"))


def skeleton(word: str) -> str:
    """Consonant skeleton: HANUMANTHAIAH, HANUMANTHIAH and HANUMANTHA all give 'HNMNT'."""
    w = re.sub(r"[^A-Z]", "", word.upper())
    if not w:
        return ""
    for a, b in _DIGRAPHS:
        w = w.replace(a, b)
    head, tail = w[0], re.sub(r"[AEIOUYH]", "", w[1:])
    out = head
    for ch in tail:
        if ch != out[-1]:
            out += ch
    return out


def name_key(name: str) -> tuple[str, ...]:
    return tuple(skeleton(w) for w in normalize_text(name).split() if not w.isdigit())


def are_variants(a: str, b: str) -> bool:
    """Two payee names that are spellings of the same name (not just similar names)."""
    if a == b:
        return True
    na, nb = normalize_text(a), normalize_text(b)
    ka, kb = name_key(na), name_key(nb)
    if not ka or ka != kb or len("".join(ka)) < 3:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= 0.8


def word_variant(word: str, name: str) -> bool:
    """A spoken/short form of one word of the name: 'hanumantha' ~ 'Hanumanthaiah'."""
    w = normalize_text(word)
    if len(w) < 4:
        return False
    sk = skeleton(w)
    for part in normalize_text(name).split():
        if part == w:
            return True
        if len(sk) >= 3 and skeleton(part) == sk and SequenceMatcher(None, w, part).ratio() >= 0.75:
            return True
    return False


# ---------------------------------------------------------------------------- features
def _grams(word: str, sizes: tuple[int, ...] = (3, 4)) -> list[str]:
    w = f"#{word}#"
    return [f"g:{w[i : i + n]}" for n in sizes for i in range(len(w) - n + 1)]


def features(text: str, *, grams: bool = True, weight: float = 1.0) -> Counter[str]:
    out: Counter[str] = Counter()
    for word in normalize_text(text).split():
        if len(word) < 2 or word.isdigit():
            continue
        out[f"w:{_stem(word)}"] += weight
        if grams and len(word) >= 4:
            for g in _grams(word):
                out[g] += 0.25 * weight
    return out


def _stem(token: str) -> str:
    for suffix in ("IES", "ES", "S", "ING"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("Y" if suffix == "IES" else "")
    return token


def category_concepts(category: str | None) -> str:
    """Words that describe a category: label, description and its everyday vocabulary."""
    from finpilot.services.retrieval import CATEGORY_WORDS

    if not category or category not in BY_ID:
        return ""
    c = BY_ID[category]
    words = [c.label, c.description, *(w for w, cats in CATEGORY_WORDS.items() if category in cats)]
    return " ".join(words)


class SparseIndex:
    """TF-IDF over hand-built features, cosine similarity."""

    def __init__(self, docs: list[Counter[str]]):
        n = len(docs) or 1
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        self.idf = {f: math.log(1 + n / c) for f, c in df.items()}
        self.vecs = [self._weigh(d) for d in docs]

    def _weigh(self, feats: Counter[str]) -> dict[str, float]:
        v = {f: math.log1p(x) * self.idf.get(f, 0.0) for f, x in feats.items() if x > 0}
        norm = math.sqrt(sum(w * w for w in v.values())) or 1.0
        return {f: w / norm for f, w in v.items() if w}

    def scores(self, query: Counter[str]) -> list[float]:
        q = self._weigh(query)
        return [sum(w * d.get(f, 0.0) for f, w in q.items()) for d in self.vecs]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion of several ranked lists of keys."""
    out: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            out[key] = out.get(key, 0.0) + 1.0 / (k + rank + 1)
    return out


class SemanticIndex:
    """Hybrid semantic search over payees (merchant name → stats with category and narrations)."""

    def __init__(
        self,
        merchants: dict[str, Any],
        narrations: dict[str, list[str]],
        router: LLMRouter | None = None,
    ):
        self.names = list(merchants)
        self.stats = merchants
        self.router = router
        self.texts: list[str] = []
        docs: list[Counter[str]] = []
        for name in self.names:
            m = merchants[name]
            cat = m.categories.most_common(1)[0][0] if m.categories else None
            tags = describe(name, " ".join(narrations.get(name, [])[:1]))
            meaning = f"{tags} {category_concepts(cat)}"
            doc = features(name, weight=2.0)
            doc.update(features(" ".join(narrations.get(name, [])[:5]), grams=False, weight=0.4))
            doc.update(features(meaning, grams=False, weight=0.8))
            if cat:
                doc[f"c:{cat}"] += 1.5
            docs.append(doc)
            self.texts.append(f"{name}. {tags}. {BY_ID[cat].label if cat in BY_ID else ''}".strip())
        self.index = SparseIndex(docs)
        self._dense: list[list[float]] | None = None

    def _query_features(self, query: str, categories: list[str]) -> Counter[str]:
        q = features(query, weight=1.0)
        for cat in categories:
            q[f"c:{cat}"] += 1.2
        return q

    def _dense_ranking(self, query: str) -> list[str]:
        if self.router is None or not self.router.embeddings_available() or not self.names:
            return []
        try:
            if self._dense is None:
                self._dense = self.router.embed(self.texts)
            (qv,) = self.router.embed([query])
        except Exception as exc:  # noqa: BLE001 - dense retrieval is an optional booster
            log.info("dense retrieval unavailable: %s", exc)
            return []
        qn = math.sqrt(sum(x * x for x in qv)) or 1.0
        sims = []
        for name, v in zip(self.names, self._dense, strict=True):
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            sims.append((sum(a * b for a, b in zip(qv, v, strict=False)) / (qn * vn), name))
        sims.sort(reverse=True)
        return [n for s, n in sims[:20] if s > 0.2]

    def search(self, query: str, *, categories: list[str] | None = None, k: int = 8) -> list[dict[str, Any]]:
        if not self.names:
            return []
        scores = self.index.scores(self._query_features(query, categories or []))
        ranked = sorted(((s, n) for s, n in zip(scores, self.names, strict=True) if s > 0.05), reverse=True)
        sparse = [n for _, n in ranked[:20]]
        dense = self._dense_ranking(query)
        by_name = dict((n, s) for s, n in ranked)
        if dense:
            fused = rrf([sparse, dense])
            order = sorted(fused, key=lambda n: -fused[n])
            mode = "hybrid (sparse + dense, RRF)"
        else:
            order = sparse
            mode = "sparse (words + char n-grams + meaning)"
        out = []
        for name in order[:k]:
            m = self.stats[name]
            cat = m.categories.most_common(1)[0][0] if m.categories else None
            out.append(
                {
                    "merchant": name,
                    "score": round(by_name.get(name, 0.0), 3),
                    "category": cat,
                    "about": describe(name) or None,
                    "transactions": m.count,
                    "retrieval": mode,
                }
            )
        return out
