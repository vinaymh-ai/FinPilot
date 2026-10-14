"""Categorization quality, transaction metadata, the auditor and the analyzer's new findings."""

import datetime as dt

import pytest

from evaluation.run_eval import HOLDOUT_MERCHANTS
from finpilot.agents.analyzer import AnalyzerAgent
from finpilot.agents.auditor import AuditorAgent
from finpilot.agents.categorizer import CategorizerAgent, CatItem
from finpilot.domain.lexicon import keyword_scores, match_merchant
from finpilot.domain.money import format_money
from finpilot.domain.normalize import (
    detect_channel,
    looks_like_person,
    merchant_key,
    normalize_text,
    payment_provider,
)
from finpilot.domain.recurring import TxnLite
from finpilot.models import Transaction
from finpilot.services import repository as repo


def rules(items):
    return CategorizerAgent(use_llm=False, use_jev=False).categorize(items)[0]


# ---------------------------------------------------------------------------- honesty of the evaluation
def test_synthetic_long_tail_merchants_never_leak_into_the_lexicon():
    leaked = [m for m in HOLDOUT_MERCHANTS if match_merchant(normalize_text(m)) is not None]
    assert leaked == []


# ---------------------------------------------------------------------------- deterministic stages
def test_phrases_hide_the_words_inside_them():
    assert (
        max(keyword_scores("NANDINI MILK PARLOUR"), key=keyword_scores("NANDINI MILK PARLOUR").get)
        == "groceries"
    )
    assert set(keyword_scores("CORNER HOUSE ICE CREAM PARLOUR")) == {"dining"}
    assert (
        max(keyword_scores("LAKME BEAUTY PARLOUR"), key=keyword_scores("LAKME BEAUTY PARLOUR").get)
        == "personal_care"
    )


@pytest.mark.parametrize(
    "description,amount,category",
    [
        ("POS 1234XXXXXX5678 SAPNA BOOK HOUSE BLR", -450, "education"),
        ("UPI/DR/1/KUMAR AUTO WORKS/SBIN/kumar@paytm/Payment", -300, "transport"),
        ("POS 1234XXXXXX5678 CORNER HOUSE ICE CREAMS BLR", -320, "dining"),
        ("UPI/DR/1/PRIYA NAIR/YESB/priyanair@oksbi/Payment", -900, "transfers"),
        ("BY TRANSFER-UPI/CR/412345678929/PRIYA SHARMA/SBIN/priya@oksbi/Payment", 2500, "transfers"),
        ("IMPS/P2A/412345678951/KARTHIK/HDFC/Lunch", -350, "transfers"),
        ("ATW-512345XXXXXX1234-S1AWBL01-BANGALORE", -5000, "cash"),
        ("ATD/AUTO DEBIT CC0XX1234/ICICI BANK CREDIT CARD", -9870, "credit_card_payment"),
        ("BY TRANSFER-NEFT*HDFC0000001*N123456789*INITECH SOLUTIONS*", 98000, "income"),
        ("ICICI PRUDENTIAL MUTUAL FUND SIP", -5000, "investments"),
        ("ICICI PRU LIFE INSURANCE PREMIUM", -5000, "insurance"),
        # every kind of bank transaction, not just merchants
        ("TAX PAYMENT-CBDT-ITNS280-ADVANCE TAX", -25000, "taxes"),
        ("NEFT-IN-CBDT INCOME TAX REFUND AY 2026-27", 18450, "income"),
        ("ATM WDL CHARGES 3 TXNS", -63, "fees"),
        ("GST @18% on Charge", -89.82, "fees"),
        ("FT - DR - 50100123456789 - SELF", -20000, "transfers"),
        ("PAYTM WALLET TOPUP", -2000, "transfers"),
        ("AMAZON PAY BALANCE LOAD", -1000, "transfers"),
        ("PAYTM CASHBACK RECEIVED", 50, "income"),
        ("Sweep Trf To: 1234567890 (FD)", -50000, "investments"),
        ("SOVEREIGN GOLD BOND SUBSCRIPTION", -6300, "investments"),
        ("EAW-416021XXXXXX1234-S1AWBL01-BANGALORE", -3000, "cash"),
        ("IB BILLPAY DR-HDFCQH-123456XXXXXX1234", -21000, "credit_card_payment"),
    ],
)
def test_generic_rules(description, amount, category):
    assert rules([CatItem(1, description, amount)])[0].category == category


def test_person_detection_needs_name_evidence_not_just_two_words():
    assert looks_like_person("UPI/DR/1/RAHUL SHARMA/UTIB/rahul.s@okicici/Payment")
    assert looks_like_person("UPI/1/HANUMANTHA/9876543210@ybl")  # phone-number handle
    assert not looks_like_person("UPI/1/GREEN TRENDS/green@icici")
    assert not looks_like_person("UPI/1/SHUBHAM ENTERPRISES/s@ybl")
    assert looks_like_person("UPI/1/SHUBHAM GUPTA/s@ybl")  # 'HUB' inside a name is not a business word


def test_icici_style_narration_uses_the_vpa_not_the_note_as_payee():
    desc = "UPI/412345678915/Payment from Ph/razorpay@icici/ICICI Bank"
    assert payment_provider(desc) == "Razorpay" and merchant_key(desc)[0] == "Razorpay"


def test_family_rules_keep_the_brand_name():
    assert merchant_key("POS 1234XXXXXX1234 THIRD WAVE COFFEE BLR")[0] == "Third Wave Coffee"
    assert merchant_key("POS CROMA RETAIL 123")[0] == "Croma"
    assert merchant_key("UPI/DR/1/SWIGGY/YESB/swiggy@ybl/Payment")[0] == "Swiggy"


def test_consistency_pass_aligns_a_merchants_uncertain_rows():
    agent = CategorizerAgent(use_llm=False, use_jev=False, history={"BLUE DOOR CAFE": "dining"})
    items = [
        CatItem(1, "POS 1234XXXXXX5678 BLUE DOOR CAFE", -400),
        CatItem(2, "UPI/1/BLUE DOOR CAFE/bd@ybl", -250),
    ]
    res, _ = agent.categorize(items)
    assert [r.category for r in res] == ["dining", "dining"]


# ---------------------------------------------------------------------------- LLM stage
def test_llm_sees_one_group_per_merchant_and_cannot_erase_evidence(fake_router_factory):
    groups = []

    def respond(messages, opts):
        import json

        payload = json.loads(messages[-1]["content"])
        groups.extend(payload["merchants"])
        return {
            "results": [
                {"id": g["id"], "category": "other", "confidence": 0.3, "reason": "unsure"}
                for g in payload["merchants"]
            ]
        }

    router = fake_router_factory(respond)
    items = [CatItem(i, f"POS 1234XXXXXX{i:04d} SHRI GANESH MEDICALS", -100.0 - i) for i in range(6)]
    res, stats = CategorizerAgent(router=router, use_jev=False).categorize(items)
    assert stats.llm_merchants == 1 and len(groups) == 1 and len(groups[0]["transactions"]) == 6
    assert all(r.category == "health" for r in res)  # an unsure model keeps the keyword evidence
    assert "could not confirm" in res[0].reason


def test_confident_lexicon_rule_yields_only_to_a_confident_model(fake_router_factory):
    router = fake_router_factory(
        lambda m, o: {"results": [{"id": 0, "category": "groceries", "confidence": 0.8, "reason": "maybe"}]}
    )
    res, _ = CategorizerAgent(router=router, use_jev=False).categorize(
        [CatItem(1, "UPI/DR/1/SWIGGY/YESB/swiggy@ybl/Payment", -300)]
    )
    assert res[0].category == "food_delivery" and "rule kept" in res[0].reason


# ---------------------------------------------------------------------------- direction & channel
@pytest.mark.parametrize(
    "description,channel",
    [
        ("UPI/DR/772691462722/RAHUL SHARMA/UTIB/rahul.s@okicici/Payment", "upi"),
        ("POS 449563XXXXXX0607 TRUFFLES BENGALURU", "card"),
        ("NACH DR BSE STAR MF GROWW SIP 12345678", "auto_debit"),
        ("NEFT CR-HDFC0000123-ACME TECHNOLOGIES PVT LTD SALARY", "neft"),
        ("MMT/IMPS/412345678901/Transfer to fam/HDFC", "imps"),
        ("NWD-412345XXXXXX1234-S1ANBL12-BANGALORE", "atm"),
        ("BIL/ONL/000123/BESCOM/ELECTRICITY", "netbanking"),
        ("CREDIT INTEREST", "bank"),
        ("CHQ DEP 000123 CLG", "cheque"),
        ("DEBIT-ACHDr HDFC0000001 HDFC BANK HOME LOAN EMI", "auto_debit"),
        ("EAW-416021XXXXXX1234-S1AWBL01-BANGALORE", "atm"),
        ("CARD TXN AT NETFLIX.COM", "card"),
    ],
)
def test_channel_detection(description, channel):
    assert detect_channel(description) == channel


def test_direction_and_channel_are_set_on_insert_update_and_exposed(db):
    repo.get_or_create_user(db, 1)
    t = Transaction(
        user_id=1,
        date=dt.date(2026, 10, 1),
        description="UPI/DR/1/SWIGGY/YESB/swiggy@ybl/Payment",
        merchant="Swiggy",
        amount=-300,
        fingerprint="a",
    )
    db.add(t)
    db.commit()
    assert (t.direction, t.channel) == ("debit", "upi")
    t.amount, t.description = 300, "NEFT CR-ACME REFUND"
    db.commit()
    assert (t.direction, t.channel) == ("credit", "neft")


def test_transactions_api_filters_by_direction_and_channel(seeded):
    from fastapi.testclient import TestClient

    from finpilot.main import create_app

    with TestClient(create_app()) as c:
        page = c.get("/api/transactions", params={"direction": "credit", "page_size": 500}).json()
        assert page["items"] and all(t["direction"] == "credit" and t["amount"] > 0 for t in page["items"])
        assert page["totals"]["debit_count"] == 0 and page["totals"]["credits"] > 0
        upi = c.get("/api/transactions", params={"channel": "upi", "page_size": 5}).json()
        assert upi["items"] and all(t["channel"] == "upi" for t in upi["items"])


# ---------------------------------------------------------------------------- auditor
def test_auditor_flags_unsupported_trails_and_passes_grounded_ones():
    good = {
        "title": "Cancel Zomato Gold",
        "message": "It costs ₹149/month.",
        "trigger": "analyzer:subscription_unused",
        "evidence": {"monthly_cost": 149, "transaction_ids": [1, 2]},
        "reasoning": ["a", "b"],
        "monthly_impact": 149,
    }
    bad = dict(
        good, trigger="analyzer:made_up", message="It costs ₹999/month.", evidence={"transaction_ids": [99]}
    )
    out = AuditorAgent(use_llm=False).audit(
        [good, bad], finding_kinds={"subscription_unused"}, has_forecast=True, ledger_ids={1, 2}
    )
    assert [r["passed"] for r in out["results"]] == [True, False]
    issues = " ".join(out["results"][1]["issues"])
    assert "no matching analyzer finding" in issues and "not in this ledger" in issues and "999" in issues


# ---------------------------------------------------------------------------- analyzer
def _ledger(as_of: dt.date) -> list[TxnLite]:
    txns, i = [], 0
    for month_back in range(4, 0, -1):
        start = (as_of.replace(day=1) - dt.timedelta(days=30 * month_back)).replace(day=1)
        for d in range(0, 28, 3):
            i += 1
            txns.append(TxnLite(i, start + dt.timedelta(days=d), -300, "Swiggy", "food_delivery"))
    for d in range(0, 30, 2):  # recent window: Swiggy and Zomato both up
        i += 1
        txns.append(TxnLite(i, as_of - dt.timedelta(days=d), -500, "Swiggy", "food_delivery"))
        i += 1
        txns.append(TxnLite(i, as_of - dt.timedelta(days=d), -250, "Zomato", "food_delivery"))
    i += 1
    txns.append(TxnLite(i, as_of - dt.timedelta(days=60), 90000, "Employer", "income"))
    return txns


def test_spike_names_the_merchants_driving_it_and_review_queue_is_reported():
    as_of = dt.date(2026, 10, 20)
    txns = _ledger(as_of)
    report = AnalyzerAgent(use_llm=False).analyze(
        txns, [], as_of=as_of, narrate=False, review_ids={txns[0].id}
    )
    spike = next(f for f in report.findings if f.kind == "category_spike")
    assert [d["merchant"] for d in spike.metrics["drivers"]][:1] == ["Swiggy"]
    assert "Swiggy" in spike.detail
    review = next(f for f in report.findings if f.kind == "review_queue")
    assert review.metrics["count"] == 1 and review.transaction_ids == [txns[0].id]


def test_indian_digit_grouping_for_rupees_only():
    assert format_money(775014.14) == "₹7,75,014.14"
    assert format_money(-28000) == "-₹28,000"
    assert format_money(775014.14, "USD") == "$775,014.14"
    assert format_money(449.16) == "₹449.16"
