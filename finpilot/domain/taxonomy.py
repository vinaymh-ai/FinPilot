"""Category taxonomy shared by agents, API and UI.

Built-in categories cover the common cases. Users can add their own ("Household help", "Pet care",
"Kids' school"): each custom category names a built-in *parent* it behaves like, so every analysis
(spending vs transfers, essential vs discretionary, refunds) treats it correctly without special
cases. Custom ids embed the owner's user id (``u12_household_help``), so they are globally unique and
one in-process registry can serve every user; `load_custom` refreshes it from the database.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    group: str  # income | essential | discretionary | financial | other
    color: str
    icon: str  # lucide icon name used by the UI
    description: str
    parent: str | None = None  # custom categories: the built-in category this one behaves like
    custom: bool = False
    user_id: int | None = None


CATEGORIES: list[Category] = [
    Category(
        "income", "Income", "income", "#22c55e", "wallet", "Salary, freelance payouts, interest, refunds"
    ),
    Category("rent", "Rent & Housing", "essential", "#6366f1", "home", "Rent, maintenance, society charges"),
    Category(
        "utilities",
        "Bills & Utilities",
        "essential",
        "#0ea5e9",
        "zap",
        "Electricity, water, gas, internet, mobile",
    ),
    Category(
        "groceries",
        "Groceries",
        "essential",
        "#14b8a6",
        "shopping-basket",
        "Supermarkets, quick-commerce, kirana",
    ),
    Category(
        "food_delivery",
        "Food Delivery",
        "discretionary",
        "#f97316",
        "bike",
        "Swiggy, Zomato and other delivery apps",
    ),
    Category("dining", "Dining Out", "discretionary", "#fb923c", "utensils", "Restaurants, cafés, bars"),
    Category(
        "transport", "Transport & Fuel", "essential", "#eab308", "car", "Cabs, metro, fuel, parking, tolls"
    ),
    Category(
        "shopping", "Shopping", "discretionary", "#ec4899", "shopping-bag", "E-commerce, fashion, electronics"
    ),
    Category(
        "subscriptions",
        "Subscriptions",
        "discretionary",
        "#a855f7",
        "repeat",
        "Streaming, software, memberships",
    ),
    Category(
        "entertainment", "Entertainment", "discretionary", "#d946ef", "ticket", "Movies, events, gaming"
    ),
    Category(
        "health", "Health & Fitness", "essential", "#ef4444", "heart-pulse", "Pharmacy, doctors, labs, gym"
    ),
    Category("education", "Education", "essential", "#3b82f6", "graduation-cap", "Courses, tuition, books"),
    Category("travel", "Travel", "discretionary", "#06b6d4", "plane", "Flights, trains, hotels"),
    Category(
        "personal_care", "Personal Care", "discretionary", "#f472b6", "sparkles", "Salon, spa, grooming"
    ),
    Category("loans", "Loans & EMIs", "financial", "#64748b", "landmark", "Loan EMIs and repayments"),
    Category(
        "credit_card_payment",
        "Credit Card Repayments",
        "financial",
        "#475569",
        "credit-card",
        "Credit-card bill settlements; distinct from purchases and person-to-person transfers",
    ),
    Category("insurance", "Insurance", "financial", "#8b5cf6", "shield", "Health, life, vehicle insurance"),
    Category(
        "investments",
        "Savings & Investments",
        "financial",
        "#10b981",
        "trending-up",
        "SIPs, mutual funds, stocks, deposits",
    ),
    Category(
        "transfers",
        "Transfers",
        "other",
        "#94a3b8",
        "arrow-left-right",
        "Person-to-person and self transfers",
    ),
    Category("cash", "Cash Withdrawal", "other", "#78716c", "banknote", "ATM withdrawals"),
    Category(
        "fees",
        "Fees & Charges",
        "essential",
        "#dc2626",
        "alert-triangle",
        "Bank charges, late fees, penalties",
    ),
    Category(
        "gifts_donations",
        "Gifts & Donations",
        "discretionary",
        "#f59e0b",
        "gift",
        "Gifts, charity, donations",
    ),
    Category("other", "Other", "other", "#a1a1aa", "circle-help", "Anything that does not fit elsewhere"),
]

BUILTIN: dict[str, Category] = {c.id: c for c in CATEGORIES}
CATEGORY_IDS: list[str] = [c.id for c in CATEGORIES]  # built-in ids (fixed vocabulary for models)
_BUILTIN_NON_SPEND = frozenset({"income", "transfers", "investments"})

# ---------------------------------------------------------------------------- custom categories
_custom: dict[str, Category] = {}
_lock = threading.Lock()
CUSTOM_ID_RE = re.compile(r"^u(\d+)_[a-z0-9_]{1,40}$")


class _Registry(Mapping[str, Category]):
    """Built-in categories plus every loaded custom category, looked up by id."""

    def __getitem__(self, key: str) -> Category:
        if key in BUILTIN:
            return BUILTIN[key]
        return _custom[key]

    def __contains__(self, key: object) -> bool:
        return key in BUILTIN or key in _custom

    def __iter__(self) -> Iterator[str]:
        yield from BUILTIN
        yield from list(_custom)

    def __len__(self) -> int:
        return len(BUILTIN) + len(_custom)


BY_ID: Mapping[str, Category] = _Registry()


def base(cat_id: str | None) -> str | None:
    """The built-in category a (possibly custom) category behaves like."""
    if cat_id in BUILTIN or cat_id is None:
        return cat_id
    c = _custom.get(cat_id)
    return c.parent if c is not None and c.parent in BUILTIN else ("other" if c is not None else cat_id)


class _GroupSet(AbstractSet[str]):
    """A live set of category ids: built-in members plus custom categories whose parent is one."""

    def __init__(self, builtin: frozenset[str]):
        self.builtin = builtin

    def __contains__(self, cat: object) -> bool:
        return cat in self.builtin or (isinstance(cat, str) and cat in _custom and base(cat) in self.builtin)

    def __iter__(self) -> Iterator[str]:
        yield from self.builtin
        yield from (c for c in list(_custom) if base(c) in self.builtin)

    def __len__(self) -> int:
        return sum(1 for _ in self)


DISCRETIONARY: AbstractSet[str] = _GroupSet(frozenset(c.id for c in CATEGORIES if c.group == "discretionary"))
ESSENTIAL: AbstractSet[str] = _GroupSet(frozenset(c.id for c in CATEGORIES if c.group == "essential"))
# Spending that is not consumption: excluded from "spend" totals and spike detection
NON_SPEND: AbstractSet[str] = _GroupSet(_BUILTIN_NON_SPEND)


def make_custom(
    cat_id: str,
    label_: str,
    parent: str,
    *,
    user_id: int,
    color: str | None = None,
    icon: str | None = None,
    description: str | None = None,
) -> Category:
    p = BUILTIN.get(parent) or BUILTIN["other"]
    return Category(
        cat_id,
        label_,
        p.group,
        color or p.color,
        icon or p.icon,
        description or f"Your category (behaves like {p.label})",
        parent=p.id,
        custom=True,
        user_id=user_id,
    )


def register(categories: list[Category], *, user_id: int | None = None) -> None:
    """Replace one user's custom categories in the registry (or add the given ones)."""
    with _lock:
        if user_id is not None:
            for cid in [k for k, c in _custom.items() if c.user_id == user_id]:
                del _custom[cid]
        for c in categories:
            _custom[c.id] = c


def unregister(cat_id: str) -> None:
    with _lock:
        _custom.pop(cat_id, None)


def load_custom(session: Any, user_id: int | None = None) -> list[Category]:
    """Refresh custom categories from the database (one user, or everyone at startup)."""
    from sqlalchemy import select

    from finpilot.models import CustomCategory

    stmt = select(CustomCategory)
    if user_id is not None:
        stmt = stmt.where(CustomCategory.user_id == user_id)
    cats = [
        make_custom(
            r.id,
            r.label,
            r.parent,
            user_id=r.user_id,
            color=r.color,
            icon=r.icon,
            description=r.description,
        )
        for r in session.scalars(stmt)
    ]
    register(cats, user_id=user_id)
    if user_id is None:
        with _lock:
            for cid in [k for k in _custom if k not in {c.id for c in cats}]:
                del _custom[cid]
    return cats


def custom_for(user_id: int | None) -> list[Category]:
    return [c for c in list(_custom.values()) if c.user_id == user_id]


def categories_for(user_id: int | None) -> list[Category]:
    """Built-in categories followed by this user's own."""
    return [*CATEGORIES, *custom_for(user_id)]


def valid_for(cat_id: str | None, user_id: int | None) -> bool:
    if cat_id in BUILTIN:
        return True
    c = _custom.get(cat_id or "")
    return c is not None and c.user_id == user_id


def slugify(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_")[:32] or "category"


def label(cat_id: str | None) -> str:
    return BY_ID[cat_id].label if cat_id in BY_ID else "Uncategorised"


def transaction_kind(category: str | None, amount: float) -> str:
    category = base(category)
    if category in {"transfers", "credit_card_payment", "investments", "cash", "income"}:
        return {"investments": "investment", "cash": "withdrawal"}.get(category, category)
    if not category or category == "other":
        return "unknown"
    return "refund" if amount > 0 else "expense"


def is_spending_flow(category: str | None, amount: float) -> bool:
    """Only identified refunds offset spending; an unexplained credit is not a negative expense."""
    return category not in NON_SPEND and (amount < 0 or base(category) not in {None, "other"})


def as_dicts(user_id: int | None = None) -> list[dict]:
    return [asdict(c) for c in categories_for(user_id)]
