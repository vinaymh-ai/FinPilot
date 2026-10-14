"""Reproduce opaque payment descriptors, missing entity filters, and ungrounded chat claims."""

import datetime as dt
import json
import math

import numpy as np
import pytest

from finpilot.agents.analyzer import AnalyzerAgent
from finpilot.agents.categorizer import CategorizerAgent, CatItem
from finpilot.agents.chat import ChatAgent
from finpilot.agents.coordinator import Coordinator
from finpilot.agents.forecaster import ForecastEngine
from finpilot.domain.recurring import TxnLite, detect_recurring
from finpilot.llm.client import LLMError
from finpilot.llm.grounding import check_grounding
from finpilot.models import AgentRun, Goal, Transaction
from finpilot.services import repository as repo
from finpilot.services.ingestion import ParsedRow, import_rows
from finpilot.services.toolkit import FinanceToolkit

KIWI = "UPI/Gokiwi Tec/XXbdpg@kotakpa/ICICI Cred/KOTAK MAHI/111111111111/TESTREF"
CRED = "UPI/CRED Club/XXclub@axisb/payment on/AXIS BANK/222222222222/TESTREF"


def test_provider_narrations_reach_reasoning_and_keep_different_purposes_separate(
    fake_router_factory, fake_jev_factory
):
    def classify(messages, opts):
        payload = json.loads(messages[-1]["content"])
        assert payload["currency"] == "INR"
        groups = payload["merchants"]
        assert len(groups) == 2
        assert groups[0]["transactions"][0]["description"] == CRED
        assert groups[0]["transactions"][0]["amount"] == -2349
        return {
            "results": [
                {
                    "id": groups[0]["id"],
                    "category": "credit_card_payment",
                    "confidence": 0.91,
                    "reason": "Bill-paying service and bank narration suggest a card bill settlement",
                },
                {
                    "id": groups[1]["id"],
                    "category": "shopping",
                    "confidence": 0.88,
                    "reason": "Narration identifies an online purchase",
                },
            ]
        }

    router = fake_router_factory(classify)
    _, jev = fake_jev_factory({}, categorizer_escalate_to_llm=False)
    items = [
        CatItem(1, CRED, -2349, dt.date(2026, 8, 7)),
        CatItem(2, CRED.replace("payment on", "online purchase"), -20961, dt.date(2026, 8, 7)),
    ]
    results, stats = CategorizerAgent(router=router, jev=jev).categorize(items)
    assert [r.category for r in results] == ["credit_card_payment", "shopping"]
    assert all(r.source == "llm" for r in results)
    assert stats.llm_merchants == 2 and len(router.calls) == 1


def test_greeting_uses_llm_without_financial_tools_or_jev(ledger, fake_router_factory, fake_jev_factory):
    router = fake_router_factory(lambda m, o: "Hi! What would you like to explore about your finances?")
    fake, jev = fake_jev_factory({"intent": lambda s, k: "overview"})
    answer = ChatAgent(router=router, jev=jev).answer(ledger, "hi")
    assert answer["model"] != "rules" and answer["trace"] == [] and answer["route"] is None
    assert len(router.calls) == 1 and router.calls[0]["tools"] is None and fake.requests == []
    assert "balance" not in answer["answer"].lower()


def test_llm_narrates_exact_merchant_total_when_jev_is_unavailable(ledger, fake_router_factory):
    def respond(messages, opts):
        payload = json.loads(messages[-1]["content"])
        if "question" in payload and "tools" in payload:
            return {"calls": [{"tool": "get_spending", "args": {"merchant": "Rahul", "period": "all_time"}}]}
        if "items" in payload:
            return {"results": [{"id": i["id"], "supported": True} for i in payload["items"]]}
        assert payload["retrieval"][0]["args"]["merchant"] == "Rahul"
        return "You sent ₹300.75 to Rahul across 2 recorded transactions."

    router = fake_router_factory(respond)
    answer = ChatAgent(router=router, use_jev=False).answer(ledger, "How much did I send to Rahul?")
    assert answer["model"] != "rules" and "₹300.75" in answer["answer"]
    assert len(router.calls) == 3 and answer["guard"]["ok"]
    assert answer["trace"][0]["result"]["total"] == 300.75


def test_borderline_supported_verdict_requires_independent_check(
    ledger, fake_router_factory, fake_jev_factory
):
    def respond(messages, opts):
        payload = json.loads(messages[-1]["content"])
        return (
            {"results": [{"id": i["id"], "supported": True} for i in payload["items"]]}
            if "items" in payload
            else "You sent ₹300.75 to Rahul in 2 transactions."
        )

    router = fake_router_factory(respond)
    _, jev = fake_jev_factory(
        {
            "support": lambda s, k: {
                "supported": 0.57,
                "unsupported_claim": 0.39,
                "contradiction": 0.04,
                "generic": 0,
            }
        }
    )
    answer = ChatAgent(router=router, jev=jev).answer(ledger, "How much did I send to Rahul?")
    assert answer["model"] != "rules" and len(router.calls) == 2
    assert answer["guard"]["label"] == "supported" and answer["guard"]["ok"]


def test_rejected_llm_answer_preserves_exact_total(ledger, fake_router_factory):
    router = fake_router_factory(lambda m, o: "You spent ₹999.99 at Netflix.")
    answer = ChatAgent(router=router, use_jev=False).answer(ledger, "How much did I send to Rahul?")
    assert answer["model"] == "rules" and "Netflix" not in answer["answer"]
    assert answer["trace"][0]["result"]["total"] == 300.75 and len(router.calls) == 3


def test_planner_retains_reasoned_provider_classification(ledger):
    row = ledger.rows[0]
    row.description, row.category, row.category_source, row.category_reason = (
        CRED,
        "credit_card_payment",
        "llm",
        "LLM: bill-paying platform",
    )
    assert not Coordinator._needs_classification(row)


def test_semantic_rejection_gets_one_grounded_rephrasing_attempt(
    ledger, fake_router_factory, fake_jev_factory
):
    drafts = iter(["You paid ₹300.75 to Netflix.", "You sent ₹300.75 to Rahul in 2 transactions."])
    router = fake_router_factory(lambda m, o: next(drafts))
    _, jev = fake_jev_factory(
        {"support": lambda s, k: "unsupported_claim" if "Netflix" in s["draft"] else "supported"}
    )
    answer = ChatAgent(router=router, jev=jev).answer(ledger, "How much did I send to Rahul?")
    assert len(router.calls) == 2 and answer["guard"]["ok"]
    assert "Netflix" not in answer["answer"] and answer["model"] != "rules"


def test_complex_tool_loop_cannot_use_history_as_evidence(ledger, fake_router_factory, fake_jev_factory):
    def respond(messages, opts):
        if messages[-1].get("role") == "tool":
            return "You paid ₹80.10 to Rahul."
        return {
            "tool_calls": [{"id": "bad-filter", "name": "get_spending", "arguments": '{"period":"all_time"}'}]
        }

    router = fake_router_factory(respond)
    _, jev = fake_jev_factory({"support": lambda s, k: "contradiction"})
    answer = ChatAgent(router=router, jev=jev).answer(
        ledger,
        "Explain how much I sent to Rahul",
        [{"role": "assistant", "content": "You paid ₹80.10 to Rahul."}],
    )
    assert answer["model"] == "rules" and answer["trace"][0]["result"]["total"] == 300.75


@pytest.fixture
def ledger(db):
    user = repo.get_or_create_user(db)
    rows = [
        ParsedRow(
            dt.date(2026, 9, 10), None, "UPI/Rahul Rao/rahul@oksbi/Payment", -100.25, category="transfers"
        ),
        ParsedRow(
            dt.date(2026, 10, 11), None, "UPI/Rahul Rao/rahul@oksbi/Payment", -200.50, category="transfers"
        ),
        ParsedRow(
            dt.date(2026, 10, 11), None, "UPI/Ramesh/9000000000@ybl/Payment", -999, category="transfers"
        ),
        ParsedRow(
            dt.date(2026, 10, 12),
            None,
            "UPI/Fresh Grocery/fresh@oksbi",
            -80.10,
            balance=50000,
            category="groceries",
        ),
    ]
    import_rows(db, user, rows, source="csv")
    return FinanceToolkit(db)


def test_payment_provider_is_separate_from_purpose():
    agent = CategorizerAgent(use_llm=False, use_jev=False, memory={"CRED": "transfers"})
    results, _ = agent.categorize([CatItem(1, KIWI, -5000), CatItem(2, CRED, -7000)])
    assert results[0].merchant == "Kiwi" and results[0].category == "credit_card_payment"
    assert results[1].merchant == "CRED" and results[1].category == "other"
    assert results[1].confidence < 0.7 and results[1].source == "review"


def test_payment_provider_uses_specific_purpose_without_spreading_to_other_payments():
    results, _ = CategorizerAgent(use_llm=False, use_jev=False).categorize(
        [
            CatItem(1, "UPI/CRED Club/club@axisb/RENT PAYMENT", -10000),
            CatItem(2, "UPI/CRED Club/club@axisb/Payment on", -10000),
            CatItem(3, "UPI/Razorpay/pay@axisb/SWIGGY ORDER", -450),
        ]
    )
    assert [r.category for r in results] == ["rent", "other", "food_delivery"]


def test_abbreviated_payee_is_suggested_but_not_silently_counted(ledger):
    import_rows(
        ledger.session,
        ledger.user,
        [ParsedRow(dt.date(2026, 10, 12), None, "UPI/Rahul Ratu/payee@bank", -2000)],
        source="csv",
    )
    kit = FinanceToolkit(ledger.session)
    result = kit.search_transactions(query="Rahul Ratusaria")
    assert result["count"] == 0
    assert result["possible_matches"][0]["statement_name"].lower() == "rahul ratu"
    answer = ChatAgent(use_llm=False, use_jev=False).answer(
        kit, "check for Rahul Ratusaria in my transactions"
    )
    assert "not counted as confirmed matches" in answer["answer"]


def test_full_name_in_compact_bank_remark_matches_the_abbreviated_display(ledger):
    import_rows(
        ledger.session,
        ledger.user,
        [ParsedRow(dt.date(2026, 10, 12), None, "UPI/RAHUL RATU/payee@bank/rahulratusaria", -2000)],
        source="csv",
    )
    kit = FinanceToolkit(ledger.session)
    result = kit.search_transactions(query="Rahul Ratusaria")
    assert result["count"] == 1 and result["transactions"][0]["amount"] == -2000
    assert kit.get_spending(merchant="Rahul Ratusaria", period="all_time")["total"] == 2000


def test_merchant_total_does_not_match_creditor_text_in_another_payees_narration(ledger):
    import_rows(
        ledger.session,
        ledger.user,
        [
            ParsedRow(dt.date(2026, 10, 12), None, KIWI, -33500.35, category="credit_card_payment"),
            ParsedRow(dt.date(2026, 10, 12), None, CRED, -2349.40, category="credit_card_payment"),
            ParsedRow(
                dt.date(2026, 10, 12),
                None,
                "UPI/Razorpay/pay@axisb/SWIGGY ORDER",
                -450,
                category="food_delivery",
            ),
        ],
        source="csv",
    )
    kit = FinanceToolkit(ledger.session)
    spending = kit.get_spending(merchant="CRED", period="all_time")
    assert spending["total"] == 2349.40 and spending["count"] == 1
    assert spending["matched_merchants"] == ["CRED"]
    assert kit.get_spending(merchant="Swiggy", period="all_time")["total"] == 450


def test_relative_periods_use_the_current_calendar_not_the_last_statement_date(ledger):
    ledger.__dict__["balance_info"] = (50000, dt.date(2025, 8, 9))
    start, end, _ = ledger.period_range("this_month")
    assert start == ledger.today.replace(day=1) and end == ledger.today
    overview = ledger.get_overview()
    assert overview["last_30_days"]["end"] == ledger.today.isoformat()
    assert overview["last_30_days"]["spend"] == 80.10
    assert overview["last_30_days"]["cash_outflows"] == 1279.60
    assert overview["last_30_days"]["cash_outflow_count"] == 3
    assert overview["last_30_days"]["spending_outflow_count"] == 1
    ledger.user.balance_source = None
    assert ledger.get_overview()["balance"] is None


def test_date_only_filters_are_not_interpreted_as_a_merchant(ledger):
    answer = ChatAgent(use_llm=False, use_jev=False).answer(
        ledger, "How much did I spend from 2026-10-01 to 2026-10-13?"
    )
    assert answer["trace"][0]["args"]["merchant"] is None
    assert answer["trace"][0]["result"]["total"] == 80.10


def test_pronoun_in_explanation_is_not_a_merchant_filter(ledger):
    filters = ChatAgent._lookup_filters(
        ledger, "Why is my spending high, and which recorded transactions account for it?"
    )
    assert filters["merchant"] is None


def test_tool_loop_repairs_rounded_money_from_existing_evidence(
    ledger, fake_router_factory, fake_jev_factory
):
    def respond(messages, opts):
        if "A previous draft failed validation" in messages[-1]["content"]:
            return "You spent ₹80.10 on groceries."
        if messages[-1]["role"] == "tool":
            return "You spent ₹80 on groceries."
        return {"tool_calls": [{"id": "spend", "name": "get_spending", "arguments": '{"period":"all_time"}'}]}

    router = fake_router_factory(respond)
    _, jev = fake_jev_factory({"support": lambda s, k: "supported"})
    answer = ChatAgent(router=router, jev=jev).answer(ledger, "Explain my spending")
    assert answer["model"] != "rules" and answer["answer"] == "You spent ₹80.10 on groceries."
    assert len(router.calls) == 3 and len(answer["trace"]) == 1


@pytest.mark.parametrize(
    "phrase,period",
    [
        ("last week", "last_week"),
        ("this week", "this_week"),
        ("yesterday", "yesterday"),
        ("today", "today"),
        ("last year", "last_year"),
    ],
)
def test_common_relative_dates_are_never_silently_changed_to_all_history(ledger, phrase, period):
    answer = ChatAgent(use_llm=False, use_jev=False).answer(ledger, f"How much did I send to Rahul {phrase}?")
    assert answer["trace"][0]["args"]["merchant"] == "Rahul"
    assert answer["trace"][0]["args"]["period"] == period


def test_person_name_and_recurring_cadence_do_not_prove_purpose():
    agent = CategorizerAgent(use_llm=False, use_jev=False)
    items = [
        CatItem(i, "UPI/Rahul Rao/rahul@oksbi", -1500, dt.date(2026, m, 5))
        for i, m in enumerate(range(1, 5), start=1)
    ]
    results, _ = agent.categorize(items)
    # A person-to-person payment is a transfer by kind, but its purpose stays unconfirmed (review).
    assert all(r.category == "transfers" and r.confidence < 0.7 for r in results)
    assert all("purpose isn't stated" in r.reason for r in results)
    lite = [
        TxnLite(r.key, it.date, it.amount, r.merchant, r.category)
        for it, r in zip(items, results, strict=True)
    ]
    assert all(r.kind != "subscription" for r in detect_recurring(lite))


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), 1.5, -1, 0.4])
def test_invalid_or_weak_model_confidence_cannot_replace_unknown(confidence, fake_router_factory):
    router = fake_router_factory(
        lambda m, o: {"results": [{"id": 0, "category": "shopping", "confidence": confidence}]}
    )
    result, _ = CategorizerAgent(router=router, use_jev=False).categorize(
        [CatItem(1, "POS QZX COMPANY", -100)]
    )
    assert result[0].category == "other"


def test_merchant_lookup_keeps_entity_even_when_jev_predicts_transfer(ledger, fake_jev_factory):
    _, jev = fake_jev_factory(
        {"intent": lambda s, k: "spending_lookup", "spend_category": lambda s, k: "transfers"}
    )
    answer = ChatAgent(use_llm=False, jev=jev).answer(ledger, "how much is have sent to rahul?")
    result = answer["trace"][0]
    assert result["args"]["merchant"] == "rahul" and result["args"]["category"] is None
    assert result["result"]["total"] == 300.75 and result["result"]["count"] == 2
    assert "300.75" in answer["answer"] and "999" not in answer["answer"]


def test_merchant_search_and_followup_do_not_turn_into_all_spending(ledger):
    agent = ChatAgent(use_llm=False, use_jev=False)
    answer = agent.answer(ledger, "check for Rahul Rao in my transactions")
    assert answer["trace"][0]["tool"] == "search_transactions"
    assert answer["trace"][0]["result"]["count"] == 2
    followup = agent.answer(ledger, "Rahul?", [{"role": "user", "content": "How much have I sent to Rahul?"}])
    assert followup["trace"][0]["args"]["query"] == "Rahul"
    missing = agent.answer(ledger, "How much did I pay to Missing Person?")
    assert missing["trace"][0]["result"]["count"] == 0
    assert "No outgoing transactions" in missing["answer"]


def test_exact_dates_name_boundaries_and_search_truncation(ledger):
    assert ledger.get_spending(merchant="Ram", period="all_time")["count"] == 0
    answer = ChatAgent(use_llm=False, use_jev=False).answer(
        ledger, "How much did I pay to Rahul in September 2026?"
    )
    assert answer["trace"][0]["result"]["total"] == 100.25
    search = ledger.search_transactions(query="Rahul", limit=1)
    assert search["count"] == 2 and search["returned_count"] == 1 and search["truncated"]
    assert not search["covers_all_recorded_transactions_in_period"]
    assert ledger.call("get_spending", {"period": "some_time"}).get("error")
    assert ledger.call("get_spending", {"category": "not_a_category"}).get("error")
    assert ledger.call("search_transactions", []).get("error")


def test_no_tool_subscription_hallucination_is_discarded(ledger, fake_router_factory):
    router = fake_router_factory(
        lambda m, o: "You have 11 subscriptions including Netflix and Spotify for ₹1,693/month."
    )
    answer = ChatAgent(router=router, use_jev=False).answer(ledger, "What subscriptions do I have?")
    assert "Netflix" not in answer["answer"] and "1,693" not in answer["answer"]
    assert answer["trace"][0]["tool"] == "get_subscriptions"
    assert answer["trace"][0]["result"]["count_active"] == 0


def test_numeric_and_claim_guards_reject_instead_of_showing_a_warning():
    agent = ChatAgent(use_jev=False)
    with pytest.raises(LLMError):
        agent._guard(
            {
                "answer": "You paid ₹500 to Netflix.",
                "trace": [{"tool": "get_spending", "result": {"total": 500, "merchant": "Rahul"}}],
            }
        )
    assert not check_grounding("You spent ₹5.", {"total": 500}, strict=True)[0]
    assert not check_grounding("You spent ₹510.", {"total": 500}, strict=True)[0]
    facts = {"summary": "Statement covers 2026-08-07 to 2026-10-10", "total": 500}
    assert check_grounding("You spent ₹500 from 2026-08-07 to 2026-10-10.", facts, strict=True)[0]
    assert not check_grounding("You spent ₹7.", facts, strict=True)[0]
    assert not check_grounding("You spent ₹500 on 2026-08-08.", facts, strict=True)[0]


def test_real_import_isolated_from_starter_data_and_old_analysis(db, seeded):
    Coordinator(use_llm=False, use_jev=False).run_pipeline()
    assert repo.latest_forecast(db, 1) is not None
    user = db.get(repo.User, 1)
    import_rows(
        db,
        user,
        [ParsedRow(dt.date(2026, 10, 12), None, "My Actual Grocery Store", -123.45, balance=4321)],
        source="csv",
    )
    kit = FinanceToolkit(db)
    assert len(kit.rows) == 1 and kit.rows[0].description == "My Actual Grocery Store"
    assert kit.goals == []
    assert kit.balance_info == (4321.0, dt.date(2026, 10, 12))
    assert db.query(Transaction).count() > 500  # sample rows are preserved, not deleted
    assert repo.latest_forecast(db, 1) is None and repo.latest_insights(db, 1) == []
    assert (
        kit.get_subscriptions()["count_active"] == 0
    )  # old sample subscriptions are not current commitments


def test_rerun_preserves_manual_labels_and_does_not_reuse_its_old_model_guess(ledger):
    row = ledger.rows[0]
    row.category, row.category_source = "gifts_donations", "user"
    ledger.session.commit()
    result = Coordinator(use_llm=False, use_jev=False).run_pipeline(recategorize=True)
    assert result["status"] == "succeeded"
    assert ledger.session.get(Transaction, row.id).category == "gifts_donations"


def test_planner_repairs_legacy_gateway_labels_without_overwriting_manual_corrections(db):
    user = repo.get_or_create_user(db)
    import_rows(db, user, [ParsedRow(dt.date(2026, 10, 12), None, CRED, -100, balance=10000)], source="csv")
    row = db.query(Transaction).first()
    row.category, row.category_source, row.category_confidence = "transfers", "rule", 0.95
    db.commit()
    result = Coordinator(use_llm=False, use_jev=False).run_pipeline()
    db.expire_all()
    assert result["summary"]["plan"][0] == "categorize"
    assert db.query(Transaction).first().category == "other"


def test_failed_forecast_stops_advice_and_marks_run_failed(ledger, monkeypatch):
    def broken(self, session, state):
        raise ValueError("simulated bad forecast")

    monkeypatch.setattr(Coordinator, "_forecast", broken)
    result = Coordinator(use_llm=False, use_jev=False).run_pipeline()
    run = ledger.session.get(AgentRun, result["run_id"])
    assert result["status"] == "failed"
    assert "advisor" not in {s.agent for s in run.steps}
    assert repo.latest_forecast(ledger.session, 1) is None


def test_sparse_history_future_transactions_and_invalid_forecast_inputs():
    as_of = dt.date(2026, 10, 12)
    txns = [
        TxnLite(1, as_of, -100, "Shop", "shopping"),
        TxnLite(2, as_of + dt.timedelta(days=1), -999999, "Future", "shopping"),
    ]
    engine = ForecastEngine(txns, [], balance=10000, as_of=as_of, n_paths=10)
    assert engine.L == 1 and engine.T.sum() == -100
    out, events = engine.run()
    assert engine.summarize(out, events)["data_quality"]["limited_history"]
    report = AnalyzerAgent(use_llm=False, use_jev=False).analyze(txns, [], as_of=as_of)
    assert report.data_quality["history_days"] == 1
    assert report.windows["last_30d_spend"] == 100
    with pytest.raises(ValueError):
        ForecastEngine([], [], balance=math.inf, as_of=as_of)


def test_same_day_shocks_are_accumulated_instead_of_overwritten():
    as_of = dt.date(2026, 10, 12)
    engine = ForecastEngine(
        [TxnLite(1, as_of, -100, "One-off", "shopping")],
        [],
        balance=10000,
        as_of=as_of,
        n_paths=50,
        exclude_ids={1},
    )
    counts = np.random.default_rng(engine.seed + 1).poisson(engine.shock_rate * engine.H, size=engine.n_paths)
    assert np.allclose(engine._shocks.sum(axis=1), -100 * counts)


def test_unknown_credit_does_not_reduce_actual_spending(ledger):
    import_rows(
        ledger.session,
        ledger.user,
        [ParsedRow(dt.date(2026, 10, 12), None, "Unknown sender credit", 9000, category="other")],
        source="csv",
    )
    kit = FinanceToolkit(ledger.session)
    assert kit.get_overview()["last_30_days"]["spend"] == 80.10
    report = AnalyzerAgent(use_llm=False, use_jev=False).analyze(kit.txns, [], as_of=kit.as_of)
    assert report.windows["last_30d_spend"] == 80.10


def test_unknown_balance_does_not_produce_an_affordability_claim_or_goal_forecast(ledger):
    ledger.user.balance_source = "derived"
    ledger.session.commit()
    result = Coordinator(use_llm=False, use_jev=False).run_pipeline()
    assert "forecast" in result["summary"]["plan"] and "advise" in result["summary"]["plan"]
    forecast = repo.latest_forecast(ledger.session, 1).payload
    assert forecast["projection_type"] == "cashflow_change"
    assert forecast["prob_overdraft"] is None and forecast["start_balance"] is None
    assert forecast["goals"] == []
    answer = ChatAgent(use_llm=False, use_jev=False).answer(ledger, "Can I afford a ₹2000 phone?")
    assert "verified" in answer["answer"]
    assert "Yes" not in answer["answer"] and "It's a stretch" not in answer["answer"]


def test_goal_archival_and_manual_spend_type_invalidate_previous_forecast(ledger):
    goal = Goal(
        user_id=1,
        name="My goal",
        target_amount=10000,
        saved_amount=1000,
        deadline=ledger.as_of + dt.timedelta(days=90),
        type_source="user",
        spend_at_deadline=False,
    )
    ledger.session.add(goal)
    ledger.session.commit()
    Coordinator(use_llm=False, use_jev=False).run_pipeline()
    assert repo.latest_forecast(ledger.session, 1) is not None
    goal.spend_at_deadline = True
    ledger.session.commit()
    assert repo.latest_forecast(ledger.session, 1) is None
    Coordinator(use_llm=False, use_jev=False).run_pipeline()
    assert repo.latest_forecast(ledger.session, 1) is not None
    goal.status = "archived"
    ledger.session.commit()
    assert repo.latest_forecast(ledger.session, 1) is None
