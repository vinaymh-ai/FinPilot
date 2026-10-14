"""Custom categories, categorization rules, bulk review and the category audit trail."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, delete, func, select
from sqlalchemy.orm import Session

from finpilot.api import serializers as ser
from finpilot.api.deps import current_user_id, get_db
from finpilot.domain import taxonomy
from finpilot.domain.normalize import payment_provider
from finpilot.domain.rules import UserRule
from finpilot.models import CategoryEvent, CategoryRule, CustomCategory, MerchantMemory, Transaction
from finpilot.schemas import BulkCategorize, CategoryCreate, CategoryUpdate, RuleBody
from finpilot.services import repository as repo

router = APIRouter()
MAX_CUSTOM = 60
MAX_RULES = 300


def _check_category(db: Session, uid: int, category: str | None) -> str:
    repo.custom_categories(db, uid)
    if not category or not taxonomy.valid_for(category, uid):
        raise HTTPException(422, f"unknown category {category}")
    return category


def _category_dict(c: taxonomy.Category, stats: dict[str, tuple[int, float]]) -> dict[str, Any]:
    n, spent = stats.get(c.id, (0, 0.0))
    return {
        "id": c.id,
        "label": c.label,
        "group": c.group,
        "color": c.color,
        "icon": c.icon,
        "description": c.description,
        "parent": c.parent,
        "parent_label": taxonomy.label(c.parent) if c.parent else None,
        "custom": c.custom,
        "transactions": n,
        "spent": round(spent, 2),
    }


def _rule_dict(r: CategoryRule) -> dict[str, Any]:
    rule = UserRule.from_row(r)
    return {
        "id": r.id,
        "category": r.category,
        "category_label": taxonomy.label(r.category),
        "terms": list(r.terms or []),
        "field": r.field,
        "match": r.match,
        "direction": r.direction,
        "min_amount": r.min_amount,
        "max_amount": r.max_amount,
        "channel": r.channel,
        "priority": r.priority,
        "enabled": r.enabled is not False,
        "source": r.source,
        "hits": r.hits or 0,
        "description": rule.describe(taxonomy.label(r.category)),
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


# ---------------------------------------------------------------------------- categories
@router.get("/categories")
def list_categories(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    repo.custom_categories(db, uid)
    scope = repo.transaction_scope(db, uid)
    stats = {
        cat: (int(n), float(-(spent or 0)))
        for cat, n, spent in db.execute(
            select(
                Transaction.category,
                func.count(),
                func.sum(case((Transaction.amount < 0, Transaction.amount), else_=0)),
            )
            .where(scope)
            .group_by(Transaction.category)
        ).all()
    }
    rules = db.scalars(
        select(CategoryRule)
        .where(CategoryRule.user_id == uid)
        .order_by(CategoryRule.priority, CategoryRule.id)
    )
    review = db.scalar(
        select(func.count())
        .select_from(Transaction)
        .where(
            scope,
            Transaction.category_source.not_in(("user", "import", "user_rule")),
            (Transaction.category.is_(None))
            | (Transaction.category == "other")
            | (Transaction.category_confidence < 0.7),
        )
    )
    return {
        "categories": [_category_dict(c, stats) for c in taxonomy.categories_for(uid)],
        "rules": [_rule_dict(r) for r in rules],
        "needs_review": int(review or 0),
        "parents": [{"id": c.id, "label": c.label, "group": c.group} for c in taxonomy.CATEGORIES],
    }


@router.post("/categories", status_code=201)
def create_category(
    body: CategoryCreate, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    repo.custom_categories(db, uid)
    if body.parent not in taxonomy.BUILTIN:
        raise HTTPException(422, "parent must be a built-in category")
    label_ = " ".join(body.label.split())
    existing = taxonomy.categories_for(uid)
    if any(c.label.lower() == label_.lower() for c in existing):
        raise HTTPException(409, f"a category called “{label_}” already exists")
    if len(taxonomy.custom_for(uid)) >= MAX_CUSTOM:
        raise HTTPException(422, f"at most {MAX_CUSTOM} custom categories")
    base_id = f"u{uid}_{taxonomy.slugify(label_)}"
    cat_id, n = base_id, 2
    while db.get(CustomCategory, cat_id) is not None:
        cat_id, n = f"{base_id[:44]}_{n}", n + 1
    row = CustomCategory(
        id=cat_id,
        user_id=uid,
        label=label_,
        parent=body.parent,
        color=body.color,
        icon=body.icon,
        description=(body.description or "").strip() or None,
    )
    db.add(row)
    db.flush()
    repo.custom_categories(db, uid)
    rule_out, applied = None, 0
    terms = [t.strip() for t in body.keywords if t and t.strip()]
    if terms:
        rule_row = CategoryRule(
            user_id=uid, terms=terms, category=cat_id, field="any", match="contains", source="user"
        )
        db.add(rule_row)
        db.flush()
        applied = repo.apply_rule(db, uid, UserRule.from_row(rule_row))
        rule_out = _rule_dict(rule_row)
    db.commit()
    return {
        "category": _category_dict(taxonomy.BY_ID[cat_id], {}),
        "rule": rule_out,
        "applied": applied,
    }


@router.patch("/categories/{cat_id}")
def update_category(
    cat_id: str, body: CategoryUpdate, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    row = db.get(CustomCategory, cat_id)
    if row is None or row.user_id != uid:
        raise HTTPException(404, "custom category not found (built-in categories cannot be edited)")
    if body.parent is not None:
        if body.parent not in taxonomy.BUILTIN:
            raise HTTPException(422, "parent must be a built-in category")
        row.parent = body.parent
    if body.label is not None:
        label_ = " ".join(body.label.split())
        if any(c.label.lower() == label_.lower() and c.id != cat_id for c in taxonomy.categories_for(uid)):
            raise HTTPException(409, f"a category called “{label_}” already exists")
        row.label = label_
    for attr in ("color", "icon", "description"):
        value = getattr(body, attr)
        if value is not None:
            setattr(row, attr, value or None)
    db.commit()
    repo.custom_categories(db, uid)
    return {"category": _category_dict(taxonomy.BY_ID[cat_id], {})}


@router.delete("/categories/{cat_id}")
def delete_category(
    cat_id: str, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Delete a custom category. Its transactions, rules and learned merchants move to its parent."""
    row = db.get(CustomCategory, cat_id)
    if row is None or row.user_id != uid:
        raise HTTPException(404, "custom category not found (built-in categories cannot be deleted)")
    parent = row.parent if row.parent in taxonomy.BUILTIN else "other"
    moved = 0
    for t in db.scalars(
        select(Transaction).where(Transaction.user_id == uid, Transaction.category == cat_id)
    ):
        repo.set_category(
            db,
            t,
            parent,
            source=t.category_source or "user",
            confidence=t.category_confidence or 0.9,
            reason=f"Moved to {taxonomy.label(parent)}: the category “{row.label}” was deleted",
            actor="user",
        )
        moved += 1
    for r in db.scalars(
        select(CategoryRule).where(CategoryRule.user_id == uid, CategoryRule.category == cat_id)
    ):
        r.category = parent
    for m in db.scalars(
        select(MerchantMemory).where(MerchantMemory.user_id == uid, MerchantMemory.category == cat_id)
    ):
        m.category = parent
    db.delete(row)
    db.commit()
    taxonomy.unregister(cat_id)
    return {"deleted": cat_id, "moved_to": parent, "transactions": moved}


# ---------------------------------------------------------------------------- rules
def _rule_from_body(body: RuleBody, rule_id: int = 0) -> UserRule:
    return UserRule(
        id=rule_id,
        category=body.category,
        terms=[t.strip() for t in body.terms if t and t.strip()],
        field=body.field,
        match=body.match,
        direction=body.direction,
        min_amount=body.min_amount,
        max_amount=body.max_amount,
        channel=body.channel or None,
        priority=body.priority,
    )


@router.post("/category-rules/preview")
def preview_rule(
    body: RuleBody, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    """What a rule would match, before saving it."""
    _check_category(db, uid, body.category)
    rule = _rule_from_body(body)
    if not rule._norm:
        raise HTTPException(422, "add at least one word to match")
    hits = repo.rule_matches(db, uid, rule)
    changes = [
        t
        for t in hits
        if t.category != body.category and (body.include_manual or t.category_source != "user")
    ]
    return {
        "matches": len(hits),
        "would_change": len(changes),
        "total": round(sum(abs(float(t.amount)) for t in hits), 2),
        "merchants": sorted({t.merchant for t in hits})[:20],
        "sample": [ser.txn(t) for t in hits[:25]],
        "description": rule.describe(taxonomy.label(body.category)),
    }


@router.get("/category-rules")
def list_rules(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    repo.custom_categories(db, uid)
    rows = db.scalars(
        select(CategoryRule)
        .where(CategoryRule.user_id == uid)
        .order_by(CategoryRule.priority, CategoryRule.id)
    )
    return {"rules": [_rule_dict(r) for r in rows]}


@router.post("/category-rules", status_code=201)
def create_rule(
    body: RuleBody, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    _check_category(db, uid, body.category)
    terms = [t.strip() for t in body.terms if t and t.strip()]
    if not terms:
        raise HTTPException(422, "add at least one word to match")
    count = db.scalar(select(func.count()).select_from(CategoryRule).where(CategoryRule.user_id == uid)) or 0
    if count >= MAX_RULES:
        raise HTTPException(422, f"at most {MAX_RULES} rules")
    row = CategoryRule(
        user_id=uid,
        terms=terms,
        field=body.field,
        match=body.match,
        direction=body.direction,
        min_amount=body.min_amount,
        max_amount=body.max_amount,
        channel=body.channel or None,
        category=body.category,
        priority=body.priority,
        enabled=body.enabled,
        source="user",
        hits=0,
    )
    db.add(row)
    db.flush()
    applied = 0
    if body.apply_to_existing and body.enabled:
        applied = repo.apply_rule(db, uid, UserRule.from_row(row), include_manual=body.include_manual)
    db.commit()
    return {"rule": _rule_dict(row), "applied": applied}


@router.patch("/category-rules/{rule_id}")
def update_rule(
    rule_id: int, body: RuleBody, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    row = db.get(CategoryRule, rule_id)
    if row is None or row.user_id != uid:
        raise HTTPException(404, "rule not found")
    _check_category(db, uid, body.category)
    terms = [t.strip() for t in body.terms if t and t.strip()]
    if not terms:
        raise HTTPException(422, "add at least one word to match")
    row.terms, row.field, row.match = terms, body.field, body.match
    row.direction, row.min_amount, row.max_amount = body.direction, body.min_amount, body.max_amount
    row.channel, row.category, row.priority, row.enabled = (
        body.channel or None,
        body.category,
        body.priority,
        body.enabled,
    )
    db.flush()
    applied = 0
    if body.apply_to_existing and body.enabled:
        applied = repo.apply_rule(db, uid, UserRule.from_row(row), include_manual=body.include_manual)
    db.commit()
    return {"rule": _rule_dict(row), "applied": applied}


@router.delete("/category-rules/{rule_id}", status_code=204)
def delete_rule(rule_id: int, uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> None:
    """Remove a rule. Transactions it labelled keep their category until the next re-categorisation."""
    db.execute(delete(CategoryRule).where(CategoryRule.user_id == uid, CategoryRule.id == rule_id))
    db.commit()


# ---------------------------------------------------------------------------- review & audit
@router.post("/transactions/bulk")
def bulk_categorize(
    body: BulkCategorize, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Approve the suggested categories of many transactions, or move them all to one category."""
    if body.category is None and not body.approve:
        raise HTTPException(422, "give a category, or approve=true to confirm the current categories")
    if body.category is not None:
        _check_category(db, uid, body.category)
    rows = list(
        db.scalars(select(Transaction).where(Transaction.user_id == uid, Transaction.id.in_(body.ids)))
    )
    if not rows:
        raise HTTPException(404, "no matching transactions")
    changed, learned = 0, 0
    merchants: dict[str, str] = {}
    for t in rows:
        category = body.category or t.category or "other"
        reason = "Confirmed by you" if body.category is None else "Recategorised by you (bulk)"
        if repo.set_category(
            db,
            t,
            category,
            source="user",
            confidence=1.0,
            reason=reason,
            actor="review" if body.approve else "user",
        ):
            changed += 1
        if body.remember and category != "other" and not payment_provider(t.description):
            merchants[t.merchant] = category
    for merchant, category in merchants.items():
        try:
            repo.remember_correction(db, uid, merchant, category)
            learned += 1
        except ValueError:
            continue
    db.commit()
    return {"updated": len(rows), "changed": changed, "learned_merchants": learned}


@router.get("/transactions/{txn_id}/history")
def category_history(
    txn_id: int, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Every category change of one transaction: who (categorizer, rule, you, chat), when and why."""
    t = db.get(Transaction, txn_id)
    if t is None or t.user_id != uid:
        raise HTTPException(404, "transaction not found")
    events = db.scalars(
        select(CategoryEvent).where(CategoryEvent.transaction_id == txn_id).order_by(CategoryEvent.id)
    )
    return {
        "transaction": ser.txn(t),
        "events": [
            {
                "id": e.id,
                "at": e.created_at.isoformat() if e.created_at else None,
                "actor": e.actor,
                "from": e.old_category,
                "from_label": taxonomy.label(e.old_category) if e.old_category else None,
                "to": e.new_category,
                "to_label": taxonomy.label(e.new_category),
                "source": e.source,
                "confidence": e.confidence,
                "reason": e.reason,
                "run_id": e.run_id,
            }
            for e in events
        ],
    }
