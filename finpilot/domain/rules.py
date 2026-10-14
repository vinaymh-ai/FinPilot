"""User-defined categorization rules ("narration contains MAID → Household help").

A rule matches on the cleaned merchant name, the raw narration, or either; optionally narrowed by
direction (debit/credit), an amount range and the payment channel. Matching is case- and
punctuation-insensitive and never uses user-supplied regular expressions (no ReDoS surface).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from finpilot.domain.normalize import normalize_text

FIELDS = ("any", "merchant", "description")
MATCHES = ("contains", "word", "equals", "starts_with")


@dataclass
class UserRule:
    id: int
    category: str
    terms: list[str]
    field: str = "any"
    match: str = "contains"
    direction: str | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    channel: str | None = None
    priority: int = 100
    source: str = "user"
    _norm: list[str] = dc_field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._norm = [n for n in (normalize_text(t) for t in self.terms) if n]

    @classmethod
    def from_row(cls, row: Any) -> UserRule:
        return cls(
            id=row.id,
            category=row.category,
            terms=list(row.terms or []),
            field=row.field or "any",
            match=row.match or "contains",
            direction=row.direction,
            min_amount=row.min_amount,
            max_amount=row.max_amount,
            channel=row.channel,
            priority=row.priority if row.priority is not None else 100,
            source=row.source or "user",
        )

    # ------------------------------------------------------------------ matching
    def _text_matches(self, text: str) -> str | None:
        norm = normalize_text(text)
        for term in self._norm:
            if self.match == "equals" and norm == term:
                return term
            if self.match == "starts_with" and norm.startswith(term):
                return term
            if self.match == "word" and re.search(rf"(?<![A-Z0-9]){re.escape(term)}(?![A-Z0-9])", norm):
                return term
            if self.match == "contains" and term in norm:
                return term
        return None

    def matches(
        self, *, description: str, merchant: str, amount: float, channel: str | None = None
    ) -> str | None:
        """The term that matched, or None."""
        if not self._norm:
            return None
        if self.direction == "debit" and amount > 0 or self.direction == "credit" and amount <= 0:
            return None
        if self.min_amount is not None and abs(amount) < self.min_amount:
            return None
        if self.max_amount is not None and abs(amount) > self.max_amount:
            return None
        if self.channel and channel and self.channel != channel:
            return None
        if self.field in ("any", "merchant") and (hit := self._text_matches(merchant)):
            return hit
        if self.field in ("any", "description") and (hit := self._text_matches(description)):
            return hit
        return None

    def describe(self, category_label: str | None = None) -> str:
        where = {"any": "payee or narration", "merchant": "payee", "description": "narration"}[self.field]
        how = {"contains": "contains", "word": "has the word", "equals": "is", "starts_with": "starts with"}[
            self.match
        ]
        terms = " or ".join(f"“{t}”" for t in self.terms)
        parts = [f"{where} {how} {terms}"]
        if self.direction:
            parts.append(f"{self.direction}s only")
        if self.min_amount is not None and self.max_amount is not None:
            parts.append(f"amount {self.min_amount:g}–{self.max_amount:g}")
        elif self.min_amount is not None:
            parts.append(f"amount ≥ {self.min_amount:g}")
        elif self.max_amount is not None:
            parts.append(f"amount ≤ {self.max_amount:g}")
        if self.channel:
            parts.append(f"via {self.channel}")
        text = ", ".join(parts)
        return f"{text} → {category_label}" if category_label else text


def first_match(
    rules: list[UserRule], *, description: str, merchant: str, amount: float, channel: str | None = None
) -> tuple[UserRule, str] | None:
    for rule in sorted(rules, key=lambda r: (r.priority, r.id)):
        hit = rule.matches(description=description, merchant=merchant, amount=amount, channel=channel)
        if hit:
            return rule, hit
    return None
