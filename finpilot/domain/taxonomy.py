"""Category taxonomy shared by agents, API and UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    group: str  # income | essential | discretionary | financial | other
    color: str
    icon: str  # lucide icon name used by the UI
    description: str


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

CATEGORY_IDS: list[str] = [c.id for c in CATEGORIES]
BY_ID: dict[str, Category] = {c.id: c for c in CATEGORIES}

DISCRETIONARY = {c.id for c in CATEGORIES if c.group == "discretionary"}
ESSENTIAL = {c.id for c in CATEGORIES if c.group == "essential"}
# Spending that is not consumption: excluded from "spend" totals and spike detection
NON_SPEND = {"income", "transfers", "investments"}


def label(cat_id: str | None) -> str:
    return BY_ID[cat_id].label if cat_id in BY_ID else "Uncategorised"


def transaction_kind(category: str | None, amount: float) -> str:
    if category in {"transfers", "credit_card_payment", "investments", "cash", "income"}:
        return {"investments": "investment", "cash": "withdrawal"}.get(category, category)
    if not category or category == "other":
        return "unknown"
    return "refund" if amount > 0 else "expense"


def is_spending_flow(category: str | None, amount: float) -> bool:
    """Only identified refunds offset spending; an unexplained credit is not a negative expense."""
    return category not in NON_SPEND and (amount < 0 or category not in {None, "other"})


def as_dicts() -> list[dict]:
    return [asdict(c) for c in CATEGORIES]
