"""Evaluation of FinPilot's System One (Jev) decisions.

    python -m evaluation.run_eval --jev          # live: Mesh /v1/evaluate, records evaluation/cassettes/jev.json
    python -m evaluation.run_eval --jev-replay   # offline: replays the recorded cassette (no key, no cost)
    python -m evaluation.run_eval                # deterministic baselines + request-size estimates only

For every decision Jev makes in production it measures, on hand-labelled data (evaluation/jev_cases.py):

* accuracy and macro-F1 against the deterministic path it replaces or augments,
* calibration — expected calibration error (ECE) and a reliability table on P(chosen option),
* the confidence threshold, *chosen on a validation split and reported on a held-out test split*
  (tuning and scoring on the same data overstates every selective-prediction number),
* option-order consistency — mid-confidence answers re-asked with the options reversed,
* cost and latency per 1,000 decisions, next to the LLM call it replaces.

Without a key the Jev rows are reported as pending: this harness never invents model numbers.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.jev_cases import CHAT, CSV_COLUMNS, FAITHFULNESS, GOALS, merchant_descriptions
from finpilot.agents.categorizer import LLM_THRESHOLD, CategorizerAgent, CatItem
from finpilot.agents.chat import ChatAgent
from finpilot.config import Settings, get_settings
from finpilot.db import configure_database, init_db
from finpilot.decisions.costs import approx_tokens, llm_equivalent_tokens
from finpilot.decisions.jev import ChoiceAnswer, Evaluation, JevClient, JevError
from finpilot.decisions.spec import (
    RESERVE_TYPES,
    SPEND_TYPES,
    chat_questions,
    chat_state,
    csv_column_questions,
    csv_column_state,
    faithfulness_questions,
    faithfulness_state,
    goal_questions,
    goal_state,
    merchant_questions,
    merchant_state,
)
from finpilot.domain.normalize import merchant_key
from finpilot.llm.client import LLMRouter, llm_cost_usd
from finpilot.llm.grounding import check_grounding
from finpilot.services.goal_types import rules_goal_type
from finpilot.services.ingestion import _map_headers

ROOT = Path(__file__).resolve().parent.parent
CASSETTE = ROOT / "evaluation" / "cassettes" / "jev.json"
OFFLINE = LLMRouter(Settings(llm_enabled=False))
BLOCKED = {"action_request", "off_topic", "prompt_injection"}
DIRECT = {
    "afford_check",
    "spending_lookup",
    "subscriptions",
    "goal_status",
    "forecast",
    "savings_advice",
    "anomalies",
    "overview",
}
AS_OF = dt.date(2026, 9, 30)


# ============================================================================ cassette client
class CassetteJev(JevClient):
    """JevClient that records live responses to a JSON cassette, or replays them offline."""

    def __init__(self, settings: Settings, path: Path, mode: str):
        super().__init__(settings)
        self.path, self.mode = path, mode
        self.tape: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        self.misses = 0

    @property
    def configured(self) -> bool:
        return True if self.mode == "replay" else super().configured

    @staticmethod
    def key(body: dict[str, Any]) -> str:
        core = {k: body[k] for k in ("model", "state", "questions")}
        return hashlib.sha256(json.dumps(core, sort_keys=False, default=str).encode()).hexdigest()

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        k = self.key(body)
        if self.mode == "replay":
            if k not in self.tape:
                self.misses += 1
                raise JevError("request not in cassette — re-record with --jev")
            hit = self.tape[k]
            return hit["response"], int(hit.get("latency_ms", 0))
        data, latency = super()._post(body)
        self.tape[k] = {"response": data, "latency_ms": latency}
        return data, latency

    def save(self) -> None:
        if self.mode == "record":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.tape, indent=1, sort_keys=True))


def make_client(mode: str) -> CassetteJev:
    base = get_settings()
    settings = base.model_copy(
        update={"jev_enabled": True, "jev_cache_enabled": False, "jev_max_concurrency": 6}
    )
    return CassetteJev(settings, CASSETTE, "replay" if mode == "replay" else "record")


# ============================================================================ metrics
def split(n: int, salt: str) -> tuple[list[int], list[int]]:
    """Deterministic 50/50 validation/test split by hashed index (stable across runs)."""
    val, test = [], []
    for i in range(n):
        h = int(hashlib.md5(f"{salt}:{i}".encode()).hexdigest(), 16)
        (val if h % 2 == 0 else test).append(i)
    return val, test


def ece(probs: list[float], correct: list[bool], bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    table, total, err = [], len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = statistics.mean(probs[i] for i in idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        err += len(idx) / total * abs(acc - conf)
        table.append(
            {"bin": f"{lo:.1f}–{hi:.1f}", "n": len(idx), "mean_p": round(conf, 3), "accuracy": round(acc, 3)}
        )
    return round(err, 4), table


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    f1s = []
    for c in set(y_true):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred, strict=True))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return round(statistics.mean(f1s), 4) if f1s else 0.0


def acc(pairs: list[tuple[Any, Any]]) -> float | None:
    return round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None


def selective(conf: list[float], correct: list[bool], idx: list[int]) -> list[dict[str, Any]]:
    rows = []
    for t in [round(x * 0.05, 2) for x in range(0, 20)]:
        kept = [i for i in idx if conf[i] >= t]
        rows.append(
            {
                "threshold": t,
                "coverage": round(len(kept) / len(idx), 4) if idx else 0.0,
                "accuracy": round(sum(correct[i] for i in kept) / len(kept), 4) if kept else None,
            }
        )
    return rows


def pick_threshold(curve: list[dict[str, Any]], target: float, min_coverage: float = 0.2) -> float:
    """Lowest threshold whose accepted answers reach `target` accuracy on the validation split."""
    for row in curve:
        if row["accuracy"] is not None and row["accuracy"] >= target and row["coverage"] >= min_coverage:
            return row["threshold"]
    best = max(
        (r for r in curve if r["accuracy"] is not None and r["coverage"] >= min_coverage),
        key=lambda r: r["accuracy"],
    )
    return best["threshold"]


def at_threshold(conf: list[float], correct: list[bool], idx: list[int], t: float) -> dict[str, Any]:
    kept = [i for i in idx if conf[i] >= t]
    return {
        "threshold": t,
        "coverage": round(len(kept) / len(idx), 4) if idx else 0.0,
        "accuracy_accepted": round(sum(correct[i] for i in kept) / len(kept), 4) if kept else None,
        "n": len(idx),
    }


def cost_block(evals: list[Evaluation], purpose: str | None, s: Settings) -> dict[str, Any]:
    toks = [e.input_tokens for e in evals]
    lats = sorted(e.latency_ms for e in evals)
    out: dict[str, Any] = {
        "requests": len(evals),
        "mean_input_tokens": round(statistics.mean(toks)) if toks else 0,
        "jev_usd_per_1k": round(statistics.mean(toks) * s.jev_price_in_per_1m / 1e6 * 1000, 4)
        if toks
        else 0.0,
        "latency_ms_p50": lats[len(lats) // 2] if lats else None,
        "latency_ms_p95": lats[int(len(lats) * 0.95) - 1] if lats else None,
    }
    if purpose in llm_equivalent_tokens():
        tin, tout = llm_equivalent_tokens()[purpose]
        out["llm_usd_per_1k_estimated"] = round(llm_cost_usd(tin, tout, s) * 1000, 4)
    return out


# ============================================================================ datasets → requests
def merchant_jobs() -> tuple[list[dict], list[tuple[Any, dict]]]:
    cases = merchant_descriptions()
    jobs = []
    for c in cases:
        m, _ = merchant_key(c["description"])
        c["clean"] = m
        jobs.append(
            (
                merchant_state(m, c["description"], credit=c["credit"], typical_amount=c["amount"], count=1),
                merchant_questions(credit=c["credit"], typical_amount=c["amount"]),
            )
        )
    return cases, jobs


def chat_jobs() -> list[tuple[Any, dict]]:
    return [(chat_state(msg), chat_questions()) for msg, _, _ in CHAT]


def faith_jobs() -> list[tuple[Any, dict]]:
    return [(faithfulness_state(f, d), faithfulness_questions()) for f, d, _, _ in FAITHFULNESS]


def goal_jobs() -> list[tuple[Any, dict]]:
    return [
        (goal_state(n, 100000, AS_OF + dt.timedelta(days=180), AS_OF), goal_questions()) for n, _, _ in GOALS
    ]


def csv_jobs() -> list[tuple[Any, dict]]:
    return [(csv_column_state(h, s, o), csv_column_questions()) for h, s, _, o in CSV_COLUMNS]


# ============================================================================ deterministic baselines
def baselines() -> dict[str, Any]:
    cases, _ = merchant_jobs()
    agent = CategorizerAgent(router=OFFLINE, use_jev=False)
    rules = [
        agent.rule_categorize(CatItem(i, c["description"], -c["amount"] if not c["credit"] else c["amount"]))
        for i, c in enumerate(cases)
    ]
    unsure = [i for i, r in enumerate(rules) if r.confidence < LLM_THRESHOLD]
    regex = [ChatAgent._regex_intent(msg) for msg, _, _ in CHAT]
    blocked_truth = [intent in BLOCKED for _, intent, _ in CHAT]
    numeric_catch = [not check_grounding(d, f)[0] for f, d, _, _ in FAITHFULNESS]
    unfaithful = [label != "supported" for _, _, label, _ in FAITHFULNESS]
    gr = [rules_goal_type(n) for n, _, _ in GOALS]
    csv_map = [next(iter(_map_headers([h]).keys()), "ignore") for h, _, _, _ in CSV_COLUMNS]
    csv_truth = ["ignore" if r in ("reference", "ignore") else r for _, _, r, _ in CSV_COLUMNS]
    return {
        "merchants": {
            "cases": len(cases),
            "rules_unsure": len(unsure),
            "rules_accuracy_all": acc(
                [(r.category, c["category"]) for r, c in zip(rules, cases, strict=True)]
            ),
            "rules_accuracy_unsure": acc([(rules[i].category, cases[i]["category"]) for i in unsure]),
        },
        "chat": {
            "cases": len(CHAT),
            "regex_intent_accuracy": acc([(p, t) for p, (_, t, _) in zip(regex, CHAT, strict=True)]),
            "regex_macro_f1": macro_f1([t for _, t, _ in CHAT], regex),
            "regex_can_refuse": False,
            "blocked_messages": sum(blocked_truth),
            "complex_messages": sum(1 for _, t, _ in CHAT if t == "complex_reasoning"),
            "note": "The regex router can only pick one of 8 answerable intents: it cannot refuse an injection or "
            "a request to move money, and cannot tell when a question needs reasoning.",
        },
        "faithfulness": {
            "cases": len(FAITHFULNESS),
            "unfaithful": sum(unfaithful),
            "numeric_check_recall": round(
                sum(c and u for c, u in zip(numeric_catch, unfaithful, strict=True))
                / max(1, sum(unfaithful)),
                4,
            ),
            "numeric_check_false_rejects": sum(
                c and not u for c, u in zip(numeric_catch, unfaithful, strict=True)
            ),
            "unfaithful_without_numbers": sum(
                1 for _, _, label, num in FAITHFULNESS if label != "supported" and not num
            ),
            "note": "The numeric grounding check only sees invented figures; invented causes, wrong merchants, "
            "reversed trends and boilerplate pass it.",
        },
        "goals": {
            "cases": len(GOALS),
            "keyword_type_accuracy": acc([(g.goal_type, t) for g, (_, t, _) in zip(gr, GOALS, strict=True)]),
            "keyword_spend_accuracy": acc(
                [(g.spend_at_deadline, s) for g, (_, _, s) in zip(gr, GOALS, strict=True)]
            ),
        },
        "csv_columns": {
            "cases": len(CSV_COLUMNS),
            "synonym_mapper_accuracy": acc(list(zip(csv_map, csv_truth, strict=True))),
            "unmapped": sum(
                1 for m, t in zip(csv_map, csv_truth, strict=True) if m == "ignore" and t != "ignore"
            ),
        },
    }


def request_estimates(s: Settings) -> dict[str, Any]:
    """Offline cost estimate from the exact request bodies production would send (≈4 chars/token)."""
    client = JevClient(s)
    out = {}
    for purpose, jobs in (
        ("merchant_category", merchant_jobs()[1]),
        ("chat_route", chat_jobs()),
        ("faithfulness", faith_jobs()),
        ("goal_type", goal_jobs()),
        ("csv_column", csv_jobs()),
    ):
        toks = [approx_tokens(client.body(st, qs)) for st, qs in jobs]
        mean = statistics.mean(toks)
        row: dict[str, Any] = {
            "estimated_input_tokens": round(mean),
            "jev_usd_per_1k_estimated": round(mean * s.jev_price_in_per_1m / 1e6 * 1000, 4),
        }
        if purpose in llm_equivalent_tokens():
            tin, tout = llm_equivalent_tokens()[purpose]
            row["llm_usd_per_1k_estimated"] = round(llm_cost_usd(tin, tout, s) * 1000, 4)
        out[purpose] = row
    return out


# ============================================================================ Jev measurements
def _answers(evals: list[Any], name: str) -> list[ChoiceAnswer | None]:
    return [e.answers.get(name) if isinstance(e, Evaluation) else None for e in evals]


def eval_merchants(client: CassetteJev, s: Settings) -> dict[str, Any]:
    cases, jobs = merchant_jobs()
    evals = client.evaluate_many(jobs, agent="eval", purpose="merchant_category")
    ans = _answers(evals, "category")
    ok_idx = [i for i, a in enumerate(ans) if a is not None]
    truth = [c["category"] for c in cases]
    pred = [a.choice if a else "other" for a in ans]
    p_choice = [a.p(a.choice) if a else 0.0 for a in ans]
    conf = [a.confidence if a else 0.0 for a in ans]
    correct = [p == t for p, t in zip(pred, truth, strict=True)]
    agent = CategorizerAgent(router=OFFLINE, use_jev=False)
    rules = [
        agent.rule_categorize(CatItem(i, c["description"], c["amount"] if c["credit"] else -c["amount"]))
        for i, c in enumerate(cases)
    ]
    unsure = [i for i in ok_idx if rules[i].confidence < LLM_THRESHOLD]
    val, test = split(len(cases), "merchants")
    val, test = [i for i in val if i in ok_idx], [i for i in test if i in ok_idx]
    curve_val = selective(conf, correct, val)
    t_star = pick_threshold(curve_val, target=0.95)

    def policy(idx: list[int], t: float) -> float | None:  # production: Jev if confident, else rules
        return acc(
            [((pred[i] if conf[i] >= t and pred[i] != "other" else rules[i].category), truth[i]) for i in idx]
        )

    e, table = ece([p_choice[i] for i in ok_idx], [correct[i] for i in ok_idx])
    # order consistency on the mid-confidence band
    mid = [i for i in ok_idx if 0.5 <= conf[i] < 0.9]
    flips, merged_ok, orig_ok = 0, 0, 0
    if mid:
        rev_jobs = [(jobs[i][0], {"category": jobs[i][1]["category"].reversed()}) for i in mid]
        rev = client.evaluate_many(rev_jobs, agent="eval", purpose="merchant_category:order_check")
        for i, r in zip(mid, rev, strict=True):
            if not isinstance(r, Evaluation):
                continue
            b = r["category"]
            flips += b.choice != pred[i]
            keys = list(jobs[i][1]["category"].criteria)
            merged = max(keys, key=lambda k: ans[i].p(k) + b.p(k))
            merged_ok += merged == truth[i]
            orig_ok += pred[i] == truth[i]
    return {
        "cases": len(cases),
        "answered": len(ok_idx),
        "accuracy_all": acc([(pred[i], truth[i]) for i in ok_idx]),
        "macro_f1_all": macro_f1([truth[i] for i in ok_idx], [pred[i] for i in ok_idx]),
        "rules_accuracy_all": acc([(rules[i].category, truth[i]) for i in ok_idx]),
        "jev_accuracy_where_rules_unsure": acc([(pred[i], truth[i]) for i in unsure]),
        "rules_accuracy_where_rules_unsure": acc([(rules[i].category, truth[i]) for i in unsure]),
        "rules_unsure": len(unsure),
        "ece": e,
        "reliability": table,
        "threshold_selection": {
            "target_accuracy": 0.95,
            "chosen_on_validation": t_star,
            "validation_curve": curve_val,
            "test_at_chosen": at_threshold(conf, correct, test, t_star),
            "test_at_default": at_threshold(conf, correct, test, s.jev_accept_confidence),
            "test_policy_accuracy_at_chosen": policy(test, t_star),
            "test_rules_only_accuracy": acc([(rules[i].category, truth[i]) for i in test]),
        },
        "order_check": {
            "mid_band": len(mid),
            "flipped": flips,
            "accuracy_original": round(orig_ok / len(mid), 4) if mid else None,
            "accuracy_after_averaging": round(merged_ok / len(mid), 4) if mid else None,
        },
        "errors": [
            {"merchant": cases[i]["merchant"], "true": truth[i], "jev": pred[i], "p": round(p_choice[i], 3)}
            for i in ok_idx
            if not correct[i]
        ][:15],
        "cost": cost_block([e for e in evals if isinstance(e, Evaluation)], "merchant_category", s),
    }


def eval_chat(client: CassetteJev, s: Settings) -> dict[str, Any]:
    jobs = chat_jobs()
    evals = client.evaluate_many(jobs, agent="eval", purpose="chat_route")
    ok = [i for i, e in enumerate(evals) if isinstance(e, Evaluation)]
    truth = [t for _, t, _ in CHAT]
    ans = _answers(evals, "intent")
    pred = [a.choice if a else "?" for a in ans]
    conf = [a.confidence if a else 0.0 for a in ans]
    p_choice = [a.p(a.choice) if a else 0.0 for a in ans]
    correct = [p == t for p, t in zip(pred, truth, strict=True)]
    t_route = s.jev_route_confidence
    handled = [i for i in ok if pred[i] in DIRECT | BLOCKED and conf[i] >= t_route]
    blocked_truth = [i for i in ok if truth[i] in BLOCKED]
    refused = [i for i in ok if pred[i] in BLOCKED and conf[i] >= t_route]
    slot_pairs: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for i in ok:
        for slot, want in CHAT[i][2].items():
            qname = {"category": "spend_category"}.get(slot, slot)
            a = evals[i].answers[qname]
            got = a.choice if a.confidence >= s.jev_accept_confidence else None
            got = None if got in ("none", "other") else got
            slot_pairs[slot].append((got, want))
    e, table = ece([p_choice[i] for i in ok], [correct[i] for i in ok])
    regex = [ChatAgent._regex_intent(m) for m, _, _ in CHAT]
    per_intent = {
        intent: acc([(pred[i], truth[i]) for i in ok if truth[i] == intent]) for intent in sorted(set(truth))
    }
    return {
        "cases": len(CHAT),
        "intent_accuracy": acc([(pred[i], truth[i]) for i in ok]),
        "intent_macro_f1": macro_f1([truth[i] for i in ok], [pred[i] for i in ok]),
        "regex_intent_accuracy": acc([(regex[i], truth[i]) for i in ok]),
        "per_intent_accuracy": per_intent,
        "slot_accuracy": {k: acc(v) for k, v in slot_pairs.items()},
        "ece": e,
        "reliability": table,
        "routing_policy": {
            "route_confidence": t_route,
            "answered_without_llm": round(len(handled) / len(ok), 4) if ok else None,
            "accuracy_of_those": acc([(pred[i], truth[i]) for i in handled]),
            "escalated_to_llm": round(1 - len(handled) / len(ok), 4) if ok else None,
            "unsafe_recall": round(sum(1 for i in blocked_truth if i in refused) / len(blocked_truth), 4)
            if blocked_truth
            else None,
            "false_refusals": sum(1 for i in refused if truth[i] not in BLOCKED),
        },
        "errors": [
            {"message": CHAT[i][0], "true": truth[i], "jev": pred[i], "conf": round(conf[i], 3)}
            for i in ok
            if not correct[i]
        ],
        "cost": cost_block([e for e in evals if isinstance(e, Evaluation)], "chat_route", s),
    }


def eval_faithfulness(client: CassetteJev, s: Settings) -> dict[str, Any]:
    jobs = faith_jobs()
    evals = client.evaluate_many(jobs, agent="eval", purpose="faithfulness")
    ok = [i for i, e in enumerate(evals) if isinstance(e, Evaluation)]
    labels = [label for _, _, label, _ in FAITHFULNESS]
    ans = _answers(evals, "support")
    p_sup = [a.p("supported") if a else 0.0 for a in ans]
    unfaithful = [label != "supported" for label in labels]
    numeric = [not check_grounding(d, f)[0] for f, d, _, _ in FAITHFULNESS]
    t = s.jev_guard_confidence
    jev_reject = [p < t for p in p_sup]
    combined = [n or j for n, j in zip(numeric, jev_reject, strict=True)]

    def pr(flags: list[bool]) -> dict[str, Any]:
        tp = sum(f and u for f, u in zip((flags[i] for i in ok), (unfaithful[i] for i in ok), strict=True))
        fp = sum(
            f and not u for f, u in zip((flags[i] for i in ok), (unfaithful[i] for i in ok), strict=True)
        )
        fn = sum(
            (not f) and u for f, u in zip((flags[i] for i in ok), (unfaithful[i] for i in ok), strict=True)
        )
        return {
            "recall_unfaithful": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "false_rejects": fp,
        }

    return {
        "cases": len(FAITHFULNESS),
        "label_accuracy": acc([(ans[i].choice, labels[i]) for i in ok]),
        "guard_threshold": t,
        "numeric_only": pr(numeric),
        "jev_only": pr(jev_reject),
        "numeric_plus_jev": pr(combined),
        "auc_p_supported": _auc([p_sup[i] for i in ok], [not unfaithful[i] for i in ok]),
        "cost": cost_block([e for e in evals if isinstance(e, Evaluation)], None, s),
    }


def eval_goals(client: CassetteJev, s: Settings) -> dict[str, Any]:
    jobs = goal_jobs()
    evals = client.evaluate_many(jobs, agent="eval", purpose="goal_type")
    ok = [i for i, e in enumerate(evals) if isinstance(e, Evaluation)]
    spend_pred = []
    for i in ok:
        t, n = evals[i]["goal_type"], evals[i]["spend_at_deadline"]
        if n.confidence >= s.jev_guard_confidence:
            spend_pred.append(n.noul >= 0.5)
        else:
            spend_pred.append(t.choice in SPEND_TYPES and t.choice not in RESERVE_TYPES)
    return {
        "cases": len(GOALS),
        "type_accuracy": acc([(evals[i]["goal_type"].choice, GOALS[i][1]) for i in ok]),
        "spend_accuracy": acc(list(zip(spend_pred, [GOALS[i][2] for i in ok], strict=True))),
        "noul_auc": _auc([evals[i]["spend_at_deadline"].noul for i in ok], [GOALS[i][2] for i in ok]),
        "cost": cost_block([e for e in evals if isinstance(e, Evaluation)], None, s),
    }


def eval_csv(client: CassetteJev, s: Settings) -> dict[str, Any]:
    jobs = csv_jobs()
    evals = client.evaluate_many(jobs, agent="eval", purpose="csv_column")
    ok = [i for i, e in enumerate(evals) if isinstance(e, Evaluation)]
    return {
        "cases": len(CSV_COLUMNS),
        "role_accuracy": acc([(evals[i]["role"].choice, CSV_COLUMNS[i][2]) for i in ok]),
        "cost": cost_block([e for e in evals if isinstance(e, Evaluation)], None, s),
    }


def _auc(scores: list[float], positives: list[bool]) -> float | None:
    pos = [s for s, p in zip(scores, positives, strict=True) if p]
    neg = [s for s, p in zip(scores, positives, strict=True) if not p]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


# ============================================================================ entry point
def run(mode: str | None) -> dict[str, Any]:
    """mode: None (baselines only) | "live" (record) | "replay"."""
    s = get_settings()
    out: dict[str, Any] = {
        "model": s.jev_model,
        "baselines": baselines(),
        "request_estimates": request_estimates(s),
        "how_to_run": "MESH_API_KEY=… python -m evaluation.run_eval --jev   (records evaluation/cassettes/jev.json; "
        "afterwards --jev-replay reproduces every number offline)",
    }
    if mode is None:
        out["status"] = "pending — no Jev measurements in this run"
        return out
    tmp = tempfile.mkdtemp()
    configure_database(f"sqlite:///{os.path.join(tmp, 'jev-eval.db')}")
    init_db()
    client = make_client(mode)
    if mode == "replay" and not client.tape:
        out["status"] = "pending — no cassette recorded yet (run with --jev and a key first)"
        return out
    if mode == "live" and not client.available():
        raise SystemExit("--jev needs MESH_API_KEY (or JEV_PROVIDER=typesafe + TYPESAFE_API_KEY)")
    out.update(
        {
            "status": f"measured ({'recorded live' if mode == 'live' else 'replayed from cassette'})",
            "provider": client.provider,
            "merchants": eval_merchants(client, s),
            "chat": eval_chat(client, s),
            "faithfulness": eval_faithfulness(client, s),
            "goals": eval_goals(client, s),
            "csv_columns": eval_csv(client, s),
            "cassette_misses": client.misses,
        }
    )
    client.save()
    return out


def render(j: dict[str, Any]) -> list[str]:
    b, est = j["baselines"], j["request_estimates"]
    lines = [
        "## 6. System One decisions (Jev)",
        "",
        f"Model `{j['model']}` via Mesh `POST /v1/evaluate`. Status: **{j['status']}**.",
        "",
        "### Deterministic paths Jev replaces or augments (measured)",
        "",
        "| Decision | Deterministic path | Result |",
        "|---|---|---|",
        f"| Merchant category ({b['merchants']['cases']} hand-labelled merchants) | lexicon + keywords | "
        f"{b['merchants']['rules_accuracy_all'] * 100:.1f}% overall; **{b['merchants']['rules_accuracy_unsure'] * 100:.1f}%** "
        f"on the {b['merchants']['rules_unsure']} it is unsure about (these go to Jev) |",
        f"| Chat routing ({b['chat']['cases']} messages) | regex intent router | {b['chat']['regex_intent_accuracy'] * 100:.1f}% "
        f"intent accuracy; cannot refuse the {b['chat']['blocked_messages']} unsafe/off-topic messages or spot the "
        f"{b['chat']['complex_messages']} that need reasoning |",
        f"| Faithfulness ({b['faithfulness']['cases']} drafts, {b['faithfulness']['unfaithful']} unfaithful) | numeric "
        f"grounding check | catches **{b['faithfulness']['numeric_check_recall'] * 100:.0f}%** of unfaithful drafts — "
        f"{b['faithfulness']['unfaithful_without_numbers']} contain no wrong number at all |",
        f"| Goal type ({b['goals']['cases']} goal names) | keyword table | type {b['goals']['keyword_type_accuracy'] * 100:.0f}%, "
        f"spend-vs-keep {b['goals']['keyword_spend_accuracy'] * 100:.0f}% |",
        f"| CSV columns ({b['csv_columns']['cases']} headers) | synonym list | {b['csv_columns']['unmapped']} headers "
        f"unrecognised |",
        "",
        "### Cost per 1,000 decisions (request sizes measured from the real request bodies; ≈4 chars/token)",
        "",
        "| Decision | Input tokens / request | Jev $ / 1k | LLM it replaces, $ / 1k (est.) |",
        "|---|---|---|---|",
    ]
    for p, r in est.items():
        llm = f"${r['llm_usd_per_1k_estimated']:.4f}" if "llm_usd_per_1k_estimated" in r else "— (new check)"
        lines.append(
            f"| {p} | ~{r['estimated_input_tokens']} | ${r['jev_usd_per_1k_estimated']:.4f} | {llm} |"
        )
    lines.append("")
    if "merchants" not in j:
        lines += [f"Jev accuracy, calibration and thresholds: **pending** — {j['how_to_run']}.", ""]
        return lines
    m, c, f, g, cv = j["merchants"], j["chat"], j["faithfulness"], j["goals"], j["csv_columns"]
    ts = m["threshold_selection"]
    lines += [
        "### Jev measurements",
        "",
        "| Decision | Jev | Deterministic | Calibration (ECE) |",
        "|---|---|---|---|",
        f"| Merchant category — all | {m['accuracy_all'] * 100:.1f}% (macro-F1 {m['macro_f1_all']:.3f}) | "
        f"{m['rules_accuracy_all'] * 100:.1f}% | {m['ece']:.3f} |",
        f"| Merchant category — where rules are unsure | {m['jev_accuracy_where_rules_unsure'] * 100:.1f}% | "
        f"{m['rules_accuracy_where_rules_unsure'] * 100:.1f}% | |",
        f"| Chat intent | {c['intent_accuracy'] * 100:.1f}% (macro-F1 {c['intent_macro_f1']:.3f}) | "
        f"{c['regex_intent_accuracy'] * 100:.1f}% (regex) | {c['ece']:.3f} |",
        f"| Goal type / spend-vs-keep | {g['type_accuracy'] * 100:.0f}% / {g['spend_accuracy'] * 100:.0f}% | "
        f"{b['goals']['keyword_type_accuracy'] * 100:.0f}% / {b['goals']['keyword_spend_accuracy'] * 100:.0f}% | |",
        f"| CSV column role | {cv['role_accuracy'] * 100:.0f}% | {b['csv_columns']['unmapped']} unmapped | |",
        "",
        f"**Threshold (categories):** chosen on the validation split for ≥95% accepted accuracy → "
        f"**{ts['chosen_on_validation']:.2f}**. On the held-out test split: coverage "
        f"{ts['test_at_chosen']['coverage'] * 100:.0f}%, accepted accuracy "
        f"{(ts['test_at_chosen']['accuracy_accepted'] or 0) * 100:.1f}%; production policy (Jev if confident, rules "
        f"otherwise) {(ts['test_policy_accuracy_at_chosen'] or 0) * 100:.1f}% vs rules-only "
        f"{(ts['test_rules_only_accuracy'] or 0) * 100:.1f}%.",
        "",
        f"**Order bias:** {m['order_check']['flipped']}/{m['order_check']['mid_band']} mid-confidence answers changed "
        f"when the options were reversed.",
        "",
        f"**Chat routing at route_confidence {c['routing_policy']['route_confidence']}:** "
        f"{(c['routing_policy']['answered_without_llm'] or 0) * 100:.0f}% of messages answered without an LLM "
        f"(intent correct for {(c['routing_policy']['accuracy_of_those'] or 0) * 100:.0f}% of them); unsafe/off-topic "
        f"recall {(c['routing_policy']['unsafe_recall'] or 0) * 100:.0f}%, {c['routing_policy']['false_refusals']} "
        f"false refusals.",
        "",
        f"**Faithfulness guard:** unfaithful-draft recall numeric-only "
        f"{(f['numeric_only']['recall_unfaithful'] or 0) * 100:.0f}% → numeric + Jev "
        f"{(f['numeric_plus_jev']['recall_unfaithful'] or 0) * 100:.0f}% "
        f"({f['numeric_plus_jev']['false_rejects']} good drafts rejected).",
        "",
    ]
    return lines
