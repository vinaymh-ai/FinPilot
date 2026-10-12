import datetime as dt
import json

import pytest

from finpilot.agents.advisor import AdvisorAgent
from finpilot.agents.analyzer import AnalyzerAgent
from finpilot.agents.categorizer import CategorizerAgent, CatItem
from finpilot.agents.forecaster import Adjustment, ForecasterAgent, GoalInput
from finpilot.config import Settings
from finpilot.domain.recurring import TxnLite, detect_recurring
from finpilot.llm.client import LLMRouter
from finpilot.services.synthetic import generate

OFFLINE = LLMRouter(Settings(llm_enabled=False))


@pytest.fixture(scope="module")
def pipeline_objects():
    ds = generate("young_professional", months=8, end=dt.date(2026, 10, 12), seed=23)
    items = [CatItem(i, t.description, t.amount, t.date) for i, t in enumerate(ds.transactions)]
    res, _ = CategorizerAgent(router=OFFLINE).categorize(items)
    lite = [
        TxnLite(i, t.date, t.amount, r.merchant, r.category, t.description, t.time)
        for i, (t, r) in enumerate(zip(ds.transactions, res, strict=True))
    ]
    rec = detect_recurring(lite)
    report = AnalyzerAgent(router=OFFLINE).analyze(lite, rec)
    goals = [
        GoalInput(
            i,
            g["name"],
            g["target_amount"],
            g["saved_amount"],
            dt.date.fromisoformat(g["deadline"]),
            g["priority"],
        )
        for i, g in enumerate(ds.goals)
    ]
    fa = ForecasterAgent(router=OFFLINE, n_paths=200)
    eng = fa.engine(
        lite,
        rec,
        balance=ds.closing_balance,
        as_of=ds.end_date,
        goals=goals,
        exclude_ids={a.txn_id for a in report.anomalies},
    )
    return ds, res, lite, rec, report, fa, eng


# ----------------------------------------------------------------------------- categorizer
def test_categorizer_rules_and_reasons():
    agent = CategorizerAgent(router=OFFLINE)
    cases = {
        "UPI/DR/1/SWIGGY/YESB/swiggy@ybl/Payment": ("food_delivery", "rule"),
        "POS 1234XXXXXX5678 SHRI GANESH MEDICALS": ("health", "keyword"),
        "UPI/1/PRIYA NAIR/priya@oksbi": ("transfers", "keyword"),
        "NEFT CR-HDFC0001234-ACME TECH PVT LTD SALARY": ("income", "rule"),
    }
    for desc, (cat, src) in cases.items():
        r = agent.rule_categorize(CatItem(1, desc, 50000.0 if "SALARY" in desc else -500.0))
        assert (r.category, r.source) == (cat, src), desc
        assert r.reason


def test_categorizer_is_amount_aware():
    agent = CategorizerAgent(router=OFFLINE)
    small = agent.rule_categorize(CatItem(1, "POS HOTEL ANNAPOORNA MYSORE", -450.0))
    large = agent.rule_categorize(CatItem(2, "POS HOTEL ANNAPOORNA MYSORE", -8400.0))
    assert small.category == "dining" and large.category == "travel"


def test_categorizer_memory_overrides_rules():
    agent = CategorizerAgent(router=OFFLINE, memory={"SWIGGY": "groceries"})
    r = agent.rule_categorize(CatItem(1, "UPI/DR/1/SWIGGY/YESB/swiggy@ybl/Payment", -300))
    assert (r.category, r.source) == ("groceries", "memory")


def test_categorizer_llm_only_for_uncertain_unique_merchants(fake_router_factory):
    def responder(messages, opts):
        payload = json.loads(messages[-1]["content"])
        return {
            "results": [
                {
                    "id": m["id"],
                    "category": "dining",
                    "confidence": 0.9,
                    "merchant": "Truffles",
                    "reason": "Bengaluru restaurant chain",
                }
                for m in payload["merchants"]
            ]
        }

    router = fake_router_factory(responder)
    agent = CategorizerAgent(router=router)
    items = [
        CatItem(i, f"POS 1234XXXXXX{i:04d} TRUFFLES BANGALORE", -600.0 - i, dt.date(2026, 5, 1 + i))
        for i in range(5)
    ]
    items.append(CatItem(99, "UPI/DR/1/SWIGGY/YESB/swiggy@ybl/Payment", -300, dt.date(2026, 5, 1)))
    results, stats = agent.categorize(items)
    assert len(router.calls) == 1  # five Truffles rows → ONE merchant in ONE batched call; Swiggy never sent
    assert stats.llm_merchants == 1
    assert all(r.category == "dining" and r.source == "llm" for r in results[:5])
    assert results[-1].source == "rule"


def test_categorizer_rejects_invalid_llm_categories(fake_router_factory):
    router = fake_router_factory(
        lambda m, o: {"results": [{"id": 0, "category": "crypto_yolo", "confidence": 1}]}
    )
    results, _ = CategorizerAgent(router=router).categorize([CatItem(1, "POS ZXQW LTD", -999.0)])
    assert results[0].category in {"other"} and results[0].source != "llm"


def test_categorizer_accuracy_rules_only(pipeline_objects):
    ds, res, *_ = pipeline_objects
    acc = sum(t.true_category == r.category for t, r in zip(ds.transactions, res, strict=True)) / len(res)
    assert acc > 0.9


# ----------------------------------------------------------------------------- analyzer
def test_analyzer_finds_the_planted_patterns(pipeline_objects):
    ds, _, lite, _, report, *_ = pipeline_objects
    kinds = {f.kind for f in report.findings}
    assert {"subscription_unused", "subscription_overlap", "anomaly_duplicate_charge", "anomaly_fee"} <= kinds
    unused = [s.merchant for s in report.subscriptions if s.unused]
    assert unused == ["Zomato Gold"]  # Zomato orders stopped 3 months ago, membership kept billing
    flagged = {a.txn_id for a in report.anomalies}
    truth = {i for i, t in enumerate(ds.transactions) if t.anomaly}
    assert len(flagged & truth) >= 3


def test_spike_detection_ignores_one_off_anomalies(pipeline_objects):
    *_, report, _, _ = pipeline_objects[:5] + (None, None)
    spikes = [f for f in report.findings if f.kind == "category_spike"]
    assert all(f.metrics["category"] != "shopping" or f.metrics["recent"] < 50000 for f in spikes)


# ----------------------------------------------------------------------------- forecaster
def test_forecast_is_deterministic_and_banded(pipeline_objects):
    *_, fa, eng = pipeline_objects
    a = fa.forecast(eng, narrate=False)
    b = fa.forecast(eng, narrate=False)
    assert a["end_balance"] == b["end_balance"]
    s = a["series"]
    assert all(lo <= mid <= hi for lo, mid, hi in zip(s["p10"], s["p50"], s["p90"], strict=True))
    assert {g["name"] for g in a["goals"]} == {"Goa trip with friends", "Emergency fund (6 months)"}


def test_simulation_uses_common_random_numbers(pipeline_objects):
    *_, fa, eng = pipeline_objects
    date = eng.as_of + dt.timedelta(days=20)
    sim = fa.simulate(eng, [Adjustment("one_time", 30000, date=date, label="Laptop")], search_wait=False)
    # same draws for baseline & scenario → the end-balance delta is exactly the purchase
    assert sim["delta"]["end_balance_p50"] == pytest.approx(-30000, abs=1)
    assert sim["verdict"] in {"affordable", "stretch", "not_affordable"} and sim["reasons"]


def test_cancelling_subscriptions_improves_trajectory(pipeline_objects):
    *_, fa, eng = pipeline_objects
    sim = fa.simulate(
        eng,
        [
            Adjustment("cancel_recurring", merchants=["Zomato Gold", "Prime Video"]),
            Adjustment("category_change", category="food_delivery", pct=-30),
        ],
    )
    assert sim["delta"]["end_balance_p50"] > 0
    assert sim["verdict"] == "affordable"


def test_huge_purchase_is_not_affordable_and_gets_suggestions(pipeline_objects):
    *_, fa, eng = pipeline_objects
    sim = fa.simulate(eng, [Adjustment("one_time", 250000, date=eng.as_of + dt.timedelta(days=10))])
    assert sim["verdict"] == "not_affordable"
    assert any(s["type"] in ("wait", "cut") for s in sim["suggestions"])


# ----------------------------------------------------------------------------- advisor
def test_advisor_recommendations_are_specific_and_traceable(pipeline_objects):
    *_, report, fa, eng = pipeline_objects
    forecast = fa.forecast(eng, narrate=False)
    out = AdvisorAgent(router=OFFLINE).advise(report, forecast)
    recs = out["recommendations"]
    assert recs and out["headline"]
    kinds = {r["kind"] for r in recs}
    assert "cancel_unused_subscription" in kinds and "goal_at_risk" in kinds
    for r in recs:
        assert r["trigger"].split(":")[0] in {"analyzer", "forecaster"}
        assert r["evidence"] and len(r["reasoning"]) >= 2
    assert recs == sorted(recs, key=lambda r: -r["priority"])


def test_advisor_rejects_hallucinated_numbers(pipeline_objects, fake_router_factory):
    *_, report, fa, eng = pipeline_objects
    forecast = fa.forecast(eng, narrate=False)

    def responder(messages, opts):
        payload = json.loads(messages[-1]["content"])
        items = []
        for it in payload["items"]:
            items.append(
                {
                    "id": it["id"],
                    "title": "Act now",
                    "message": "Do this to save ₹99,999 every month."
                    if it["id"] == 0
                    else it["template_message"],
                }
            )
        return {"headline": "Big week", "items": items}

    router = fake_router_factory(responder)
    out = AdvisorAgent(router=router).advise(report, forecast)
    first = out["recommendations"][0]
    assert "99,999" not in first["message"]  # invented figure rejected → template kept
    assert any("rejected" in step for step in first["reasoning"])
    assert any(r["generated_by"].startswith("mesh:") for r in out["recommendations"][1:])


def test_advisor_snoozes_dismissed_advice(pipeline_objects):
    *_, report, fa, eng = pipeline_objects
    forecast = fa.forecast(eng, narrate=False)
    agent = AdvisorAgent(router=OFFLINE)
    first = agent.advise(report, forecast)["recommendations"]
    target = next(r for r in first if r["kind"] == "cancel_unused_subscription")
    feedback = [
        {
            "fingerprint": target["fingerprint"],
            "kind": target["kind"],
            "status": "dismissed",
            "monthly_impact": target["monthly_impact"],
            "updated_at": dt.date.today(),
        }
    ]
    second = agent.advise(report, forecast, feedback=feedback)
    assert target["fingerprint"] not in {r["fingerprint"] for r in second["recommendations"]}
    assert second["suppressed"] == 1
