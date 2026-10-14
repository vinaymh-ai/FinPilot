"""Regression cases for API reasoning, missing balances and changing repayment narrations."""

import datetime as dt
import json

import numpy as np
import pytest

from finpilot.agents.analyzer import AnalyzerAgent
from finpilot.agents.categorizer import CategorizerAgent, CatItem
from finpilot.agents.chat import ChatAgent, parse_amount
from finpilot.agents.coordinator import Coordinator
from finpilot.agents.forecaster import Adjustment, ForecastEngine, ForecasterAgent
from finpilot.domain.recurring import TxnLite, detect_recurring
from finpilot.llm.client import LLMResponse
from finpilot.llm.grounding import check_grounding
from finpilot.llm.validation import validate_claims
from finpilot.models import Transaction
from finpilot.services import repository as repo
from finpilot.services.toolkit import FinanceToolkit


def obligation_ledger():
    txns = []
    for month, card in ((8, 23000.32), (9, 55000.98), (10, 33500.35)):
        prefix = "INFT " if month == 10 else ""
        txns.extend(
            [
                TxnLite(len(txns) + 1, dt.date(2026, month, 7), 94571, "Employer", "income", "Salary"),
                TxnLite(
                    len(txns) + 2,
                    dt.date(2026, month, 10),
                    -38449,
                    prefix + "Personal Loan",
                    "loans",
                    prefix + "Personal Loan XX12345 EMI",
                ),
                TxnLite(
                    len(txns) + 3,
                    dt.date(2026, month, 9),
                    -card,
                    "Kiwi" if month == 10 else "CRED",
                    "credit_card_payment",
                    "Card repayment",
                ),
            ]
        )
    txns.append(TxnLite(10, dt.date(2026, 10, 9), -30000, "Own account", "transfers"))
    return txns


def test_unknown_balance_still_runs_forecast_advisor_and_produces_auditable_advice(db):
    user = repo.get_or_create_user(db, 1)
    user.balance_source = None
    for t in obligation_ledger():
        db.add(
            Transaction(
                user_id=1,
                date=t.date,
                description=t.description or t.merchant,
                amount=t.amount,
                merchant=t.merchant,
                category=t.category,
                category_source="user",
                category_confidence=1,
                fingerprint=f"regression-{t.id}",
            )
        )
    db.commit()
    result = Coordinator(use_llm=False).run_pipeline()
    assert result["status"] == "succeeded", result
    assert result["summary"]["plan"] == ["analyze", "forecast", "advise", "audit"]
    db.expire_all()
    forecast = repo.latest_forecast(db, 1).payload
    assert forecast["projection_type"] == "cashflow_change"
    assert forecast["start_balance"] is None and forecast["prob_overdraft"] is None
    assert not forecast["goals"]
    recs = repo.latest_recommendations(db, 1)
    assert {r.kind for r in recs} >= {"review_repayments", "review_cashflow"}
    assert all(r.evidence["transaction_ids"] and r.trigger and r.reasoning for r in recs)
    assert all(r.monthly_impact == 0 for r in recs)
    assert all(t.category_source == "user" for t in repo.load_transactions(db, 1))


def test_contract_reference_links_loan_and_card_budget_keeps_all_bills():
    txns = obligation_ledger()
    txns.append(TxnLite(11, dt.date(2026, 8, 9), -2000, "CRED", "credit_card_payment"))
    series = detect_recurring(txns, dt.date(2026, 10, 10))
    loan = next(r for r in series if r.category == "loans")
    card = next(r for r in series if r.category == "credit_card_payment")
    assert loan.occurrences == 3 and set(loan.txn_ids) == {2, 5, 8}
    assert set(card.txn_ids) == {3, 6, 9, 11}
    assert card.merchant == "Monthly card repayments"
    engine = ForecastEngine(txns, series, balance=100000, as_of=dt.date(2026, 10, 10), horizon_days=30)
    baseline, _ = engine.run()
    changed, _ = engine.run([Adjustment("one_time", amount=2349.40, date=dt.date(2026, 10, 15))])
    assert np.ptp(baseline.paths[:, -1]) > 1000  # varying observed bills create uncertainty
    assert np.allclose(changed.paths[:, 4:] - baseline.paths[:, 4:], -2349.40)


def test_unknown_balance_simulation_reports_exact_effect_without_yes_no():
    txns = obligation_ledger()
    engine = ForecastEngine(
        txns, detect_recurring(txns), balance=999999, balance_known=False, as_of=dt.date(2026, 10, 10)
    )
    result = ForecasterAgent(use_llm=False).simulate(
        engine, [Adjustment("one_time", amount=2000, date=dt.date(2026, 10, 15))]
    )
    assert result["verdict"] == "needs_balance"
    assert result["delta"]["end_balance_p50"] == -2000
    assert result["delta"]["prob_overdraft"] is None
    assert result["scenario"]["required_starting_balance_p90"] >= 0
    assert not result["suggestions"]


def test_api_planner_ranks_actual_expense_instead_of_searching_question(db, fake_router_factory):
    repo.get_or_create_user(db, 1)
    for tid, amount, merchant, category in (
        (1, -60000, "Own account", "transfers"),
        (2, -53812.98, "CRED", "credit_card_payment"),
        (3, -38449, "Loan", "loans"),
    ):
        db.add(
            Transaction(
                user_id=1,
                date=dt.date(2026, 9, 7),
                description=merchant,
                amount=amount,
                merchant=merchant,
                category=category,
                category_source="user",
                fingerprint=str(tid),
            )
        )
    db.commit()

    def respond(messages, opts):
        if "query planner" in messages[0]["content"]:
            return {
                "calls": [
                    {
                        "tool": "rank_spending",
                        "args": {"group_by": "merchant", "period": "all_time", "limit": 1},
                    }
                ]
            }
        if "financial fact checker" in messages[0]["content"]:
            items = json.loads(messages[-1]["content"])["items"]
            return {
                "results": [
                    {"id": i["id"], "supported": True, "reason": "Exact recorded payment"} for i in items
                ]
            }
        return "Your largest recorded expense is CRED: ₹53,812.98 on 2026-09-07. Transfers and investments are excluded."

    router = fake_router_factory(respond)
    result = ChatAgent(router=router).answer(FinanceToolkit(db), "whats my biggest expense?")
    assert result["planner"]["status"] == "executed"
    assert result["trace"][0]["tool"] == "rank_spending"
    assert result["trace"][0]["result"]["ranked"][0]["amount"] == 53812.98
    assert "53,812.98" in result["answer"] and "60,000" not in result["answer"]
    assert result["model"] != "rules" and result["guard"]["ok"]


def test_api_categorization_also_reviews_known_rule_results(fake_router_factory):
    seen = []

    def respond(messages, opts):
        payload = json.loads(messages[-1]["content"])
        seen.extend(payload["merchants"])
        return {
            "results": [
                {
                    "id": i["id"],
                    "category": "food_delivery",
                    "confidence": 0.94,
                    "reason": "Food delivery merchant and debit narration",
                }
                for i in payload["merchants"]
            ]
        }

    router = fake_router_factory(respond)
    result, _ = CategorizerAgent(router=router).categorize(
        [CatItem(1, "UPI/SWIGGY/order", -250, dt.date(2026, 10, 10))]
    )
    assert seen and result[0].source == "llm" and result[0].category == "food_delivery"


def test_bank_history_alone_does_not_prove_subscription_unused():
    txns = [
        TxnLite(i, dt.date(2026, month, 1), -149, "Swiggy One", "subscriptions")
        for i, month in enumerate((8, 9, 10), 1)
    ]
    report = AnalyzerAgent(use_llm=False).analyze(txns, detect_recurring(txns), as_of=dt.date(2026, 10, 10))
    assert not any(s.unused for s in report.subscriptions)
    assert not any(f.kind == "subscription_unused" for f in report.findings)


def test_validator_rejects_duplicate_ids_and_string_booleans(fake_router_factory):
    router = fake_router_factory(
        lambda *_: {
            "results": [
                {"id": 0, "supported": True},
                {"id": 0, "supported": True},
                {"id": 1, "supported": "true"},
            ]
        }
    )
    verdicts = validate_claims(
        router, [("a", {"amount": 100}, "Amount 100"), ("b", {}, "All steady")], agent="advisor"
    )
    assert all(not v.ok for v in verdicts)


def test_simulation_rejects_purchase_beyond_horizon():
    engine = ForecastEngine([], [], balance=1000, as_of=dt.date(2026, 10, 10), horizon_days=30)
    with pytest.raises(ValueError, match="horizon"):
        engine.run([Adjustment("one_time", amount=100, date=dt.date(2027, 10, 10))])


def test_truncated_structured_output_gets_one_uncached_retry(fake_router_factory, monkeypatch):
    router = fake_router_factory(lambda *_: {})
    calls = []

    def respond(messages, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return LLMResponse(
                content='{"results": [',
                provider="mesh",
                model="test",
                raw={"choices": [{"finish_reason": "length"}]},
            )
        return LLMResponse(content='{"results": []}', provider="mesh", model="test")

    monkeypatch.setattr(router, "chat", respond)
    data, _ = router.chat_json([], agent="categorizer", max_tokens=1000)
    assert data == {"results": []} and len(calls) == 2
    assert calls[1]["cache"] is False and calls[1]["max_tokens"] > calls[0]["max_tokens"]


def test_second_reasoning_level_reviews_uncertain_payment_business(fake_router_factory):
    prompts = []

    def respond(messages, opts):
        prompts.append(messages[0]["content"])
        second = "Second reasoning level" in prompts[-1]
        return {
            "results": [
                {
                    "id": 0,
                    "category": "credit_card_payment" if second else "other",
                    "confidence": 0.8 if second else 0.6,
                    "reason": "Ordinary bill-paying business; repayment inferred"
                    if second
                    else "Membership fee unknown",
                }
            ]
        }

    router = fake_router_factory(respond)
    result, stats = CategorizerAgent(router=router).categorize(
        [CatItem(1, "UPI/CRED Club/cred.club@axis/payment on/AXIS BANK/TESTREF", -2349.40)]
    )
    assert len(prompts) == 2 and stats.llm_batches == 2
    assert result[0].category == "credit_card_payment" and result[0].source == "llm"


@pytest.mark.parametrize(
    "plan",
    [{"calls": [{"tool": "get_overview", "args": {}}]}, {"clarification": "What is your phone budget?"}],
)
def test_affordability_planner_cannot_skip_the_requested_simulation(db, fake_router_factory, plan):
    user = repo.get_or_create_user(db, 1)
    user.balance_source = None

    def respond(messages, opts):
        if "query planner" in messages[0]["content"]:
            return plan
        if "financial fact checker" in messages[0]["content"]:
            return {"results": [{"id": 0, "supported": True}]}
        return "A verified starting balance is needed to decide affordability."

    router = fake_router_factory(respond)
    answer = ChatAgent(router=router).answer(FinanceToolkit(db), "Can I afford a ₹2000 phone next month?")
    assert [t["tool"] for t in answer["trace"]] == ["simulate"]
    assert answer["trace"][0]["result"]["delta"]["end_balance_p50"] == -2000
    assert answer["trace"][0]["result"]["verdict"] == "needs_balance"
    assert answer["model"] != "rules" and answer["guard"]["ok"]


def test_advice_fallback_is_readable_and_does_not_dump_unrelated_tool_json(db):
    kit = FinanceToolkit(db)
    trace = [
        {"tool": "get_overview", "result": kit.get_overview()},
        {
            "tool": "get_recommendations",
            "result": {
                "recommendations": [
                    {
                        "title": "Review your loan",
                        "message": "Confirm the recorded due date.",
                        "action": "Check the loan statement",
                    }
                ]
            },
        },
    ]
    summary = ChatAgent._tool_summary(kit, trace)
    assert "Check the loan statement" in summary
    assert '"currency"' not in summary and '"balance"' not in summary


def test_numeric_checks_allow_list_positions_and_rounded_percentages_but_preserve_money():
    assert check_grounding("1. Repayments take 76% of income.", {"share": 76.04}, strict=True)[0]
    assert not check_grounding("Refund ₹301.00", {"amount": 300.75}, strict=True)[0]
    assert check_grounding("Payment on 2026‑09‑11", {"date": "2026-09-11"}, strict=True)[0]


def test_unknown_balance_caveat_cannot_be_omitted_in_generated_affordability_answer():
    from finpilot.llm.client import LLMError

    with pytest.raises(LLMError, match="verified starting balance"):
        ChatAgent(use_llm=False)._guard(
            {
                "answer": "Otherwise you can afford the phone.",
                "trace": [{"tool": "simulate", "result": {"verdict": "needs_balance"}}],
            }
        )


def test_multiple_amounts_do_not_replace_purchase_budget_with_salary():
    assert parse_amount("I earn ₹95000. Can I afford a ₹5000 phone?", unique=True) is None


def test_new_recurring_cost_starting_as_of_applies_to_first_future_day():
    as_of = dt.date(2026, 10, 10)
    engine = ForecastEngine([], [], balance=10000, as_of=as_of, horizon_days=30)
    baseline, _ = engine.run()
    changed, _ = engine.run([Adjustment("recurring", amount=100, date=as_of)])
    assert np.allclose(changed.paths - baseline.paths, -100)


def test_advice_planner_retrieves_recommendations_instead_of_only_an_overview(db, fake_router_factory):
    def respond(messages, opts):
        if "query planner" in messages[0]["content"]:
            return {"calls": [{"tool": "get_overview", "args": {}}]}
        return "No supported recommendations are stored; run the agents."

    router = fake_router_factory(respond)
    answer = ChatAgent(router=router).answer(FinanceToolkit(db), "What should I do about my repayments?")
    assert "get_recommendations" in [t["tool"] for t in answer["trace"]]


def test_truncated_plain_response_is_not_shown_as_an_answer(fake_router_factory, monkeypatch):
    from finpilot.llm.client import LLMError

    router = fake_router_factory(lambda *_: {})
    monkeypatch.setattr(
        router,
        "_call",
        lambda *args, **kwargs: LLMResponse(
            content="Here is unfinished thinking",
            provider="mesh",
            model="test",
            raw={"choices": [{"finish_reason": "length"}]},
        ),
    )
    with pytest.raises(LLMError, match="truncated"):
        router.chat([], agent="chat")


def test_monthly_forecast_net_matches_actual_expected_paths_after_a_scenario():
    as_of = dt.date(2026, 10, 1)
    engine = ForecastEngine([], [], balance=10000, as_of=as_of, horizon_days=30)
    out, events = engine.run([Adjustment("one_time", amount=2000, date=as_of + dt.timedelta(days=10))])
    forecast = engine.summarize(out, events)
    assert forecast["monthly"][0]["net"] == -2000
    assert forecast["monthly"][0]["other_projected_cashflow"] == -2000


def test_zero_safety_buffer_is_preserved_as_an_explicit_user_setting(db):
    user = repo.get_or_create_user(db, 1)
    user.safety_buffer = 0
    assert FinanceToolkit(db).engine().safety_buffer == 0


def test_internal_thinking_text_is_rejected_even_when_its_numbers_are_correct():
    from finpilot.llm.client import LLMError

    with pytest.raises(LLMError, match="Internal reasoning"):
        ChatAgent(use_llm=False)._guard(
            {"answer": "Here's a thinking process: analyze user input.", "trace": []}
        )
