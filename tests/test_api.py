import datetime as dt
import json

import pytest
from fastapi.testclient import TestClient

from finpilot.main import create_app


@pytest.fixture
def client(seeded):
    from finpilot.agents.coordinator import Coordinator

    Coordinator().run_pipeline(trigger="manual")
    with TestClient(create_app()) as c:
        yield c


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert {a["id"] for a in meta["agents"]} >= {"categorizer", "analyzer", "forecaster", "advisor", "chat"}
    assert "plan" in client.get("/api/graph").json()["mermaid"]


def test_dashboard_has_forecast_insights_and_recommendations(client):
    d = client.get("/api/dashboard").json()
    assert d["forecast"]["series"]["p50"]
    assert d["recommendations"] and d["insights"]
    assert d["overview"]["transactions"] > 500


def test_run_records_decision_trail(client):
    runs = client.get("/api/runs").json()["runs"]
    detail = client.get(f"/api/runs/{runs[0]['id']}").json()["run"]
    agents = [s["agent"] for s in detail["steps"]]
    assert agents == ["coordinator", "categorizer", "analyzer", "forecaster", "advisor"]
    rec = client.get("/api/recommendations").json()["recommendations"][0]
    full = client.get(f"/api/recommendations/{rec['id']}").json()
    assert full["recommendation"]["trigger"] and full["recommendation"]["reasoning"]
    assert full["run"]["id"] == rec["run_id"]


def test_category_correction_is_learned(client):
    page = client.get("/api/transactions", params={"q": "Truffles"}).json()
    assert page["items"]
    t = page["items"][0]
    r = client.patch(f"/api/transactions/{t['id']}", json={"category": "dining"}).json()
    assert r["learned"] and r["updated"] == page["total"]
    # new upload of the same merchant is categorised from memory
    csv_text = (
        f"date,description,amount\n{dt.date(2026, 10, 12)},POS 9999XXXXXX0000 TRUFFLES NEW BRANCH,-777\n"
    )
    up = client.post("/api/transactions/upload", files={"file": ("s.csv", csv_text, "text/csv")}).json()
    assert up["batch"]["imported"] == 1
    new = client.get("/api/transactions", params={"q": "NEW BRANCH"}).json()["items"][0]
    assert new["category"] == "dining"


def test_simulate_and_chat(client):
    sim = client.post(
        "/api/simulate",
        json={
            "adjustments": [
                {"type": "one_time", "amount": 60000, "date": "2026-11-15", "label": "Manali trip"}
            ]
        },
    ).json()
    assert sim["verdict"] in {"affordable", "stretch", "not_affordable"}
    assert len(sim["scenario"]["series"]["p50"]) == len(sim["baseline"]["series"]["p50"])
    chat = client.post("/api/chat", json={"message": "Can I afford a ₹40,000 trip next month?"}).json()
    assert [t["tool"] for t in chat["trace"]] == ["simulate"]
    assert "40,000" in chat["answer"]


def test_dismissed_recommendation_is_snoozed_on_next_run(client):
    recs = client.get("/api/recommendations").json()["recommendations"]
    target = next(r for r in recs if r["kind"] == "cancel_unused_subscription")
    client.patch(f"/api/recommendations/{target['id']}", json={"status": "dismissed"})
    assert client.post("/api/runs", json={}).json()["status"] == "succeeded"
    again = client.get("/api/recommendations").json()["recommendations"]
    assert target["fingerprint"] not in {r["fingerprint"] for r in again}


def test_goals_crud_and_projection(client):
    g = client.post(
        "/api/goals",
        json={"name": "New phone", "target_amount": 80000, "saved_amount": 5000, "deadline": "2027-02-01"},
    ).json()["goal"]
    client.post("/api/runs", json={})
    goals = client.get("/api/goals").json()["goals"]
    proj = next(x for x in goals if x["id"] == g["id"])["projection"]
    assert proj and 0 <= proj["probability"] <= 1
    assert client.delete(f"/api/goals/{g['id']}").status_code == 204


def test_digest_send_now_and_history(client):
    res = client.post("/api/digests/send-now").json()
    assert res["summary"]["digest_id"]
    d = client.get("/api/digests").json()["digests"][0]
    assert d["deliveries"]["in_app"]["status"] == "sent" and d["subject"]


def test_llm_settings_override_roundtrip(client):
    st = client.put(
        "/api/settings/llm",
        json={
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "agents": {"advisor": {"provider": "mesh", "model": "openai/gpt-4o"}},
        },
    ).json()
    assert st["overrides"]["provider"] == "groq"
    assert st["overrides"]["agents"]["advisor"]["model"] == "openai/gpt-4o"
    st = client.put("/api/settings/llm", json={"provider": "", "model": "", "agents": {"advisor": {}}}).json()
    assert "provider" not in st["overrides"]


def test_recurring_cancel_removes_from_forecast(client):
    items = client.get("/api/recurring").json()["items"]
    netflix = next(i for i in items if i["merchant"] == "Netflix")
    client.patch(f"/api/recurring/{netflix['id']}", json={"status": "cancelled"})
    client.post("/api/runs", json={})
    fc = client.get("/api/forecast").json()
    assert all(e["merchant"] != "Netflix" for e in fc["upcoming"])


def test_chat_with_llm_tool_calling(client, fake_router_factory):
    state = {"n": 0}

    def responder(messages, opts):
        state["n"] += 1
        if state["n"] == 1:
            args = json.dumps({"category": "food_delivery", "period": "last_month"})
            return {"content": None, "tool_calls": [{"id": "c1", "name": "get_spending", "arguments": args}]}
        tool_msg = next(m for m in messages if m["role"] == "tool")
        total = json.loads(tool_msg["content"])["total"]
        return f"You spent ₹{total:,.0f} on food delivery last month."

    fake_router_factory(responder)
    out = client.post("/api/chat", json={"message": "food delivery spend last month?"}).json()
    assert out["model"].startswith("mesh:")
    assert out["trace"][0]["tool"] == "get_spending"
    assert "food delivery" in out["answer"]
