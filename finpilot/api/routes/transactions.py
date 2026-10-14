"""Transactions, CSV upload, recurring series."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from finpilot.agents.coordinator import Coordinator
from finpilot.api import serializers as ser
from finpilot.api.deps import current_user_id, get_db
from finpilot.domain.taxonomy import CATEGORY_IDS
from finpilot.models import RecurringSeries, Transaction
from finpilot.schemas import RecurringUpdate, TransactionCreate, TransactionUpdate
from finpilot.services import repository as repo
from finpilot.services.ingestion import ParsedRow, StatementError, import_rows, parse_file

router = APIRouter()
log = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # PDF statements for a year can be several MB


@router.get("/transactions")
def list_transactions(
    q: str | None = None,
    category: str | None = None,
    source: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    anomalies: bool = False,
    recurring: bool = False,
    impulse: bool = False,
    review: bool = False,
    direction: str | None = Query(default=None, pattern="^(in|out|credit|debit)$"),
    channel: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    uid: int = Depends(current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    scope = repo.transaction_scope(db, uid)
    stmt = select(Transaction).where(scope)
    if review:
        stmt = stmt.where(
            or_(
                Transaction.category.is_(None),
                Transaction.category == "other",
                Transaction.category_confidence < 0.7,
            )
        )
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Transaction.description.ilike(like), Transaction.merchant.ilike(like)))
    if category:
        stmt = stmt.where(Transaction.category == category)
    if source:
        stmt = stmt.where(Transaction.category_source == source)
    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    if anomalies:
        stmt = stmt.where(Transaction.is_anomaly.is_(True))
    if recurring:
        stmt = stmt.where(Transaction.is_recurring.is_(True))
    if direction in ("in", "credit"):
        stmt = stmt.where(Transaction.amount > 0)
    elif direction in ("out", "debit"):
        stmt = stmt.where(Transaction.amount <= 0)
    if channel:
        stmt = stmt.where(Transaction.channel == channel)
    rows = list(
        db.scalars(stmt.order_by(Transaction.date.desc(), Transaction.time.desc(), Transaction.id.desc()))
    )
    if impulse:
        rows = [t for t in rows if t.tags and "impulse" in t.tags]
    total = len(rows)
    items = rows[(page - 1) * page_size : page * page_size]
    sources = dict(
        db.execute(
            select(Transaction.category_source, func.count())
            .where(scope)
            .group_by(Transaction.category_source)
        ).all()
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "totals": {
            "debits": round(sum(-float(t.amount) for t in rows if t.amount <= 0), 2),
            "credits": round(sum(float(t.amount) for t in rows if t.amount > 0), 2),
            "debit_count": sum(1 for t in rows if t.amount <= 0),
            "credit_count": sum(1 for t in rows if t.amount > 0),
        },
        "items": [ser.txn(t) for t in items],
        "sources": {str(k): v for k, v in sources.items()},
    }


@router.post("/transactions", status_code=201)
def create_transaction(
    body: TransactionCreate,
    run: bool = True,
    uid: int = Depends(current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if body.category and body.category not in CATEGORY_IDS:
        raise HTTPException(422, f"unknown category {body.category}")
    user = repo.get_or_create_user(db, uid)
    row = ParsedRow(
        body.date, body.time, body.description.strip(), round(body.amount, 2), category=body.category
    )
    batch = import_rows(db, user, [row], source="manual")
    if batch.rows_imported == 0:
        raise HTTPException(409, "duplicate transaction")
    result = Coordinator().run_pipeline(uid, trigger="manual") if run else None
    t = db.scalar(select(Transaction).where(Transaction.batch_id == batch.id))
    db.refresh(t)
    return {"transaction": ser.txn(t), "run": result}


@router.patch("/transactions/{txn_id}")
def update_transaction(
    txn_id: int, body: TransactionUpdate, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    t = db.get(Transaction, txn_id)
    if t is None or t.user_id != uid:
        raise HTTPException(404, "transaction not found")
    updated = 1
    if body.category is not None:
        if body.category not in CATEGORY_IDS:
            raise HTTPException(422, f"unknown category {body.category}")
        if body.apply_to_merchant:
            updated = repo.remember_correction(db, uid, t.merchant, body.category)
        else:
            t.category, t.category_source, t.category_confidence = body.category, "user", 1.0
            t.category_reason = "Corrected by you"
    if body.notes is not None:
        t.notes = body.notes
    db.commit()
    db.refresh(t)
    return {
        "transaction": ser.txn(t),
        "updated": updated,
        "learned": bool(body.category and body.apply_to_merchant),
    }


@router.post("/transactions/upload")
async def upload(
    file: UploadFile = File(...),
    password: str | None = Form(default=None),
    run: bool = True,
    uid: int = Depends(current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Import a bank statement: CSV/TSV/TXT, Excel (.xlsx/.xls), HTML-as-XLS or PDF (optionally password-protected)."""
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    try:
        parsed = await run_in_threadpool(parse_file, content, file.filename, password=password or None)
    except StatementError as exc:
        log.warning("statement rejected: %s", exc, extra={"event": "upload_rejected", "code": exc.code})
        raise HTTPException(
            422, {"message": str(exc), "code": exc.code, "errors": [], "filename": file.filename}
        ) from exc
    if not parsed.rows:
        log.warning(
            "no transactions found in %s", file.filename,
            extra={"event": "upload_empty", "format": parsed.format, "errors": parsed.errors[:5]},
        )  # fmt: skip
        raise HTTPException(
            422,
            {
                "message": "No transactions found in the file",
                "code": "no_rows",
                "errors": parsed.errors[:10],
                "warnings": parsed.warnings,
                "mapping": parsed.mapping,
                "format": parsed.format,
                "filename": file.filename,
            },
        )
    user = repo.get_or_create_user(db, uid)
    batch = import_rows(
        db,
        user,
        parsed.rows,
        source=parsed.format or "csv",
        filename=file.filename,
        errors=parsed.errors,
        total=parsed.total,
    )
    result = None
    if run and batch.rows_imported:
        result = await run_in_threadpool(Coordinator().run_pipeline, uid, trigger="upload")
    return {
        "batch": {
            "id": batch.id,
            "filename": batch.filename,
            "format": parsed.format,
            "total": batch.rows_total,
            "imported": batch.rows_imported,
            "duplicates": batch.rows_duplicate,
            "invalid": batch.rows_invalid,
            "errors": parsed.errors[:10],
            "warnings": parsed.warnings,
            "mapping": parsed.mapping,
            "summary": parsed.summary,
        },
        "run": result,
    }


@router.get("/recurring")
def list_recurring(uid: int = Depends(current_user_id), db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = list(db.scalars(select(RecurringSeries).where(RecurringSeries.user_id == uid)))
    items = sorted(
        [ser.recurring(r) for r in rows], key=lambda r: (r["status"] != "active", -r["monthly_cost"])
    )
    run = repo.latest_run(db, uid)
    subs = ((run.analysis or {}).get("subscriptions") if run else None) or []
    flags = {s["merchant"]: s for s in subs}
    for it in items:
        f = flags.get(it["merchant"])
        it["unused"] = bool(f and f.get("unused"))
        it["overlap_with"] = f.get("overlap_with", []) if f else []
        if f and f.get("usage_signal") and not it.get("usage_signal"):
            it["usage_signal"] = f["usage_signal"]
    active_subs = [i for i in items if i["kind"] == "subscription" and i["status"] == "active"]
    return {
        "items": items,
        "subscriptions_monthly": round(sum(i["monthly_cost"] for i in active_subs), 2),
        "subscriptions_count": len(active_subs),
    }


@router.patch("/recurring/{series_id}")
def update_recurring(
    series_id: int, body: RecurringUpdate, uid: int = Depends(current_user_id), db: Session = Depends(get_db)
) -> dict[str, Any]:
    r = db.get(RecurringSeries, series_id)
    if r is None or r.user_id != uid:
        raise HTTPException(404, "series not found")
    if body.usage:
        r.usage = body.usage
    if body.status == "cancelled":
        _, as_of = repo.current_balance(db, repo.get_or_create_user(db, uid))
        r.status, r.cancelled_on = "cancelled", as_of
    elif body.status == "active":
        r.status, r.cancelled_on = "active", None
    db.commit()
    return {"series": ser.recurring(r)}
