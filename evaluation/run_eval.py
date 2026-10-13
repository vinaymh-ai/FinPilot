"""FinPilot self-evaluation.

    python -m evaluation.run_eval              # deterministic (rules + statistics), no API cost
    python -m evaluation.run_eval --llm        # also measure the LLM layer through the configured provider (Mesh)
    python -m evaluation.run_eval --jev        # also measure System One (Jev) live; records a replay cassette
    python -m evaluation.run_eval --jev-replay # re-measure Jev offline from the recorded cassette

Writes evaluation/results/latest.json (served at /api/evaluation and shown in the UI) and
evaluation/results/REPORT.md. Test data: synthetic statements for 3 personas × N seeds with
ground-truth labels (see finpilot/services/synthetic.py), including long-tail local merchants that
are deliberately absent from the merchant lexicon, injected anomalies and planted behaviour shifts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finpilot.agents.analyzer import AnalyzerAgent  # noqa: E402
from finpilot.agents.categorizer import CategorizerAgent, CatItem  # noqa: E402
from finpilot.agents.forecaster import Adjustment, ForecasterAgent  # noqa: E402
from finpilot.config import Settings  # noqa: E402
from finpilot.domain.lexicon import match_merchant  # noqa: E402
from finpilot.domain.normalize import normalize_text  # noqa: E402
from finpilot.domain.recurring import TxnLite, detect_recurring  # noqa: E402
from finpilot.domain.stats import add_months  # noqa: E402
from finpilot.llm.client import LLMRouter, get_router  # noqa: E402
from finpilot.services.synthetic import SynthDataset, generate  # noqa: E402

PERSONAS = ["young_professional", "family", "freelancer"]
END = dt.date(2026, 9, 30)
OFFLINE = LLMRouter(Settings(llm_enabled=False))
OUT = ROOT / "evaluation" / "results"


# ============================================================================ helpers
def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    per = {}
    for c in sorted(set(y_true)):
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p != c)
        per[c] = prf(tp, fp, fn) | {"support": sum(1 for t in y_true if t == c)}
    return round(statistics.mean(v["f1"] for v in per.values()), 4), per


def datasets(seeds: list[int], months: int = 8, end: dt.date = END) -> list[SynthDataset]:
    return [generate(p, months=months, end=end, seed=s) for p in PERSONAS for s in seeds]


def categorize(ds: SynthDataset, router: LLMRouter, *, use_patterns: bool = True):
    items = [CatItem(i, t.description, t.amount, t.date) for i, t in enumerate(ds.transactions)]
    return CategorizerAgent(router=router, use_jev=False).categorize(items, use_patterns=use_patterns)


def lite_from(ds: SynthDataset, results) -> list[TxnLite]:
    return [
        TxnLite(i, t.date, t.amount, r.merchant, r.category, t.description, t.time)
        for i, (t, r) in enumerate(zip(ds.transactions, results, strict=True))
    ]


# ============================================================================ 1. categorization
def eval_categorization(
    dsets: list[SynthDataset], llm_router: LLMRouter | None, jev_client: Any = None
) -> dict[str, Any]:
    y_true, preds, seen_flags, sources = [], defaultdict(list), [], Counter()
    rows_total = 0
    llm_stats: dict[str, Any] = {}
    for ds in dsets:
        rows_total += len(ds.transactions)
        y_true += [t.true_category for t in ds.transactions]
        seen_flags += [match_merchant(normalize_text(t.description)) is not None for t in ds.transactions]
        agent = CategorizerAgent(router=OFFLINE, use_jev=False)
        items = [CatItem(i, t.description, t.amount, t.date) for i, t in enumerate(ds.transactions)]
        # ablation ladder: lexicon → +keywords/P2P/amount-aware → +recurring patterns → +LLM
        lex = []
        for it in items:
            r = agent.rule_categorize(it)
            lex.append(r.category if r.source in ("rule", "memory") else "other")
        preds["1_lexicon_only"] += lex
        res_nopat, _ = agent.categorize(items, use_patterns=False)
        preds["2_plus_keywords"] += [r.category for r in res_nopat]
        res, st = agent.categorize(items)
        preds["3_plus_patterns (deterministic)"] += [r.category for r in res]
        sources.update(r.source for r in res)
        if jev_client is not None:
            res_jev, st_jev = CategorizerAgent(router=OFFLINE, jev=jev_client).categorize(items)
            preds["4_plus_jev"] += [r.category for r in res_jev]
            llm_stats.setdefault("jev_merchants", 0)
            llm_stats["jev_merchants"] += st_jev.jev_merchants
            llm_stats["jev_input_tokens"] = llm_stats.get("jev_input_tokens", 0) + st_jev.jev_tokens
        if llm_router is not None:
            t0 = time.perf_counter()
            res_llm, st_llm = CategorizerAgent(router=llm_router, use_jev=False).categorize(items)
            preds["4_plus_llm"] += [r.category for r in res_llm]
            llm_stats.setdefault("merchants_sent", 0)
            llm_stats["merchants_sent"] += st_llm.llm_merchants
            llm_stats["batches"] = llm_stats.get("batches", 0) + st_llm.llm_batches
            llm_stats["seconds"] = round(llm_stats.get("seconds", 0) + time.perf_counter() - t0, 1)
            llm_stats["model"] = st_llm.model
    out: dict[str, Any] = {"transactions": rows_total, "datasets": len(dsets), "ablation": {}}
    naive = Counter(y_true).most_common(1)[0][0]
    out["baseline_majority_class"] = {
        "predicts": naive,
        "accuracy": round(sum(1 for y in y_true if y == naive) / len(y_true), 4),
    }
    for mode, yp in preds.items():
        acc = sum(1 for a, b in zip(y_true, yp, strict=True) if a == b) / len(y_true)
        mf1, per = macro_f1(y_true, yp)
        seen = [a == b for a, b, s in zip(y_true, yp, seen_flags, strict=True) if s]
        unseen = [a == b for a, b, s in zip(y_true, yp, seen_flags, strict=True) if not s]
        out["ablation"][mode] = {
            "accuracy": round(acc, 4),
            "macro_f1": mf1,
            "accuracy_known_merchants": round(sum(seen) / len(seen), 4) if seen else None,
            "accuracy_long_tail": round(sum(unseen) / len(unseen), 4) if unseen else None,
        }
        if mode.startswith(("3", "4")):
            out["ablation"][mode]["per_category"] = per
            errors = Counter((a, b) for a, b in zip(y_true, yp, strict=True) if a != b)
            out["ablation"][mode]["top_confusions"] = [
                {"true": a, "pred": b, "count": c} for (a, b), c in errors.most_common(8)
            ]
    out["long_tail_share"] = round(1 - sum(seen_flags) / len(seen_flags), 4)
    out["source_mix"] = {k: round(v / sum(sources.values()), 4) for k, v in sources.most_common()}
    if llm_stats:
        out["llm"] = llm_stats
    return out


# ============================================================================ 2. recurring + anomalies
def eval_detectors(dsets: list[SynthDataset]) -> dict[str, Any]:
    tp = fp = fn = 0
    a_tp = a_fp = a_fn = 0
    by_kind: Counter[str] = Counter()
    detected_kinds: Counter[str] = Counter()
    unused_hits = unused_total = 0
    for ds in dsets:
        res, _ = categorize(ds, OFFLINE)
        lite = lite_from(ds, res)
        series = detect_recurring(lite, ds.end_date)
        truth_keys = Counter(t.recurring_key for t in ds.transactions if t.recurring_key)
        eligible = {
            k
            for k, n in truth_keys.items()
            if n >= (2 if ds.recurring_truth.get(k, {}).get("cadence") in ("quarterly", "yearly") else 3)
        }
        matched = set()
        for s in series:
            keys = Counter(ds.transactions[i].recurring_key for i in s.txn_ids)
            key, cnt = keys.most_common(1)[0]
            if key and cnt / len(s.txn_ids) >= 0.6:
                tp += 1
                matched.add(key)
            else:
                fp += 1
        fn += len(eligible - matched)
        # anomalies
        report = AnalyzerAgent(router=OFFLINE, use_jev=False).analyze(lite, series, narrate=False)
        flagged = {a.txn_id for a in report.anomalies}
        truth = {i for i, t in enumerate(ds.transactions) if t.anomaly}
        a_tp += len(flagged & truth)
        a_fp += len(flagged - truth)
        a_fn += len(truth - flagged)
        for i in truth:
            by_kind[ds.transactions[i].anomaly] += 1
            if i in flagged:
                detected_kinds[ds.transactions[i].anomaly] += 1
        if ds.persona == "young_professional":
            unused_total += 1
            unused_hits += int([s.merchant for s in report.subscriptions if s.unused] == ["Zomato Gold"])
    return {
        "recurring_series": prf(tp, fp, fn),
        "anomalies": prf(a_tp, a_fp, a_fn)
        | {"recall_by_type": {k: round(detected_kinds[k] / v, 3) for k, v in by_kind.items()}},
        "unused_membership_detection": {"datasets": unused_total, "exact_match": unused_hits},
    }


# ============================================================================ 3. forecast backtest
def _flows(txns: list[TxnLite], start: dt.date, end: dt.date) -> float:
    return sum(t.amount for t in txns if start < t.date <= end)


def eval_forecast(seeds: list[int], horizons=(30, 60, 90)) -> dict[str, Any]:
    errs: dict[str, dict[int, list[float]]] = {
        m: {h: [] for h in horizons} for m in ("finpilot_p50", "naive_last_30d", "mean_last_90d")
    }
    cover: dict[int, list[bool]] = {h: [] for h in horizons}
    cat_abs, cat_act = defaultdict(float), defaultdict(float)
    cat_abs_naive = defaultdict(float)
    n_origins = 0
    incomes = []
    per_persona: dict[str, list[float]] = defaultdict(list)
    t0 = time.perf_counter()
    for persona in PERSONAS:
        for seed in seeds:
            ds = generate(persona, months=12, end=END, seed=seed)
            res, _ = categorize(ds, OFFLINE, use_patterns=False)  # per-txn rules only → no look-ahead leakage
            lite = lite_from(ds, res)
            start = min(t.date for t in lite)
            income_m = statistics.mean(
                [t.amount for t in lite if t.category == "income" and t.amount > 20000] or [1]
            )
            for k in range(5, 12):  # origins at month-ends 5..11 (need ≥4 months history, ≥1 month ahead)
                origin = (start.replace(day=1) + dt.timedelta(days=32 * k)).replace(day=1) - dt.timedelta(
                    days=1
                )
                if origin + dt.timedelta(days=max(horizons)) > END + dt.timedelta(days=0):
                    hz = [h for h in horizons if origin + dt.timedelta(days=h) <= END]
                else:
                    hz = list(horizons)
                if not hz:
                    continue
                train = [t for t in lite if t.date <= origin]
                balance0 = ds.opening_balance + sum(t.amount for t in train)
                series = detect_recurring(train, origin)
                report = AnalyzerAgent(router=OFFLINE, use_jev=False).analyze(
                    train, series, as_of=origin, narrate=False
                )
                eng = ForecasterAgent(router=OFFLINE, use_jev=False, n_paths=300).engine(
                    train,
                    series,
                    balance=balance0,
                    as_of=origin,
                    goals=[],
                    exclude_ids={a.txn_id for a in report.anomalies},
                    horizon_days=max(horizons),
                )
                out, _ = eng.run()
                # baselines on calendar months (origins are month-ends) so each contains one payday
                month_start = origin.replace(day=1)
                last30 = _flows(train, month_start - dt.timedelta(days=1), origin)
                last90 = _flows(train, add_months(month_start, -2) - dt.timedelta(days=1), origin) / 3
                n_origins += 1
                incomes.append(income_m)
                for h in hz:
                    actual = _flows(lite, origin, origin + dt.timedelta(days=h))
                    col = out.paths[:, h - 1] - balance0
                    p10, p50, p90 = np.percentile(col, [10, 50, 90])
                    errs["finpilot_p50"][h].append(abs(p50 - actual) / income_m)
                    if h == 30:
                        per_persona[persona].append(abs(p50 - actual) / income_m)
                    errs["naive_last_30d"][h].append(abs(last30 * h / 30 - actual) / income_m)
                    errs["mean_last_90d"][h].append(abs(last90 * h / 30 - actual) / income_m)
                    cover[h].append(bool(p10 <= actual <= p90))
                # category-level next-30-day variable spend (discretionary + groceries/transport)
                rates = eng.category_monthly_rates()
                rec_ids = {i for s in series for i in s.txn_ids}
                nxt = [t for t in lite if origin < t.date <= origin + dt.timedelta(days=30)]
                prev = [t for t in train if origin - dt.timedelta(days=30) < t.date and t.id not in rec_ids]
                for c in ("food_delivery", "groceries", "dining", "transport", "shopping", "entertainment"):
                    actual_c = sum(
                        -t.amount
                        for t in nxt
                        if t.category == c and t.amount < 0 and not ds.transactions[t.id].recurring_key
                    )
                    pred_c = rates.get(c, 0.0) * 30 / 30.4
                    naive_c = sum(-t.amount for t in prev if t.category == c and t.amount < 0)
                    cat_abs[c] += abs(pred_c - actual_c)
                    cat_abs_naive[c] += abs(naive_c - actual_c)
                    cat_act[c] += actual_c
    summary: dict[str, Any] = {
        "origins": n_origins,
        "seconds": round(time.perf_counter() - t0, 1),
        "metric": "mean |predicted − actual net cash-flow| as % of monthly income",
        "horizons": {},
    }
    for h in horizons:
        if not errs["finpilot_p50"][h]:
            continue
        row = {m: round(100 * statistics.mean(v[h]), 2) for m, v in errs.items()}
        row["finpilot_p50_median"] = round(100 * statistics.median(errs["finpilot_p50"][h]), 2)
        row["naive_last_30d_median"] = round(100 * statistics.median(errs["naive_last_30d"][h]), 2)
        row["p10_p90_coverage"] = round(sum(cover[h]) / len(cover[h]), 3)
        row["n"] = len(cover[h])
        summary["horizons"][str(h)] = row
    summary["error_30d_by_persona"] = {p: round(100 * statistics.mean(v), 1) for p, v in per_persona.items()}
    summary["category_wape_30d"] = {
        c: {
            "finpilot": round(cat_abs[c] / cat_act[c], 3) if cat_act[c] else None,
            "naive_last_30d": round(cat_abs_naive[c] / cat_act[c], 3) if cat_act[c] else None,
        }
        for c in cat_act
    }
    tot_f, tot_n, tot_a = sum(cat_abs.values()), sum(cat_abs_naive.values()), sum(cat_act.values())
    summary["category_wape_30d_overall"] = {
        "finpilot": round(tot_f / tot_a, 3),
        "naive_last_30d": round(tot_n / tot_a, 3),
    }
    return summary


# ============================================================================ 4. simulation consistency
def eval_simulation(seeds: list[int]) -> dict[str, Any]:
    checks = monotone = exact = 0
    order = {"affordable": 0, "stretch": 1, "not_affordable": 2}
    for persona in PERSONAS:
        for seed in seeds:
            ds = generate(persona, months=8, end=END, seed=seed)
            res, _ = categorize(ds, OFFLINE)
            lite = lite_from(ds, res)
            series = detect_recurring(lite)
            fa = ForecasterAgent(router=OFFLINE, use_jev=False, n_paths=200)
            eng = fa.engine(
                lite, series, balance=ds.closing_balance, as_of=ds.end_date, goals=[], exclude_ids=set()
            )
            last = -1
            ok = True
            for amt in (5_000, 20_000, 50_000, 100_000, 200_000, 400_000):
                sim = fa.simulate(
                    eng,
                    [Adjustment("one_time", amt, date=ds.end_date + dt.timedelta(days=15))],
                    search_wait=False,
                )
                checks += 1
                exact += int(abs(sim["delta"]["end_balance_p50"] + amt) < 1)
                v = order[sim["verdict"]]
                ok = ok and v >= last
                last = v
            monotone += int(ok)
    n = len(PERSONAS) * len(seeds)
    return {
        "scenarios": checks,
        "exact_delta_rate": round(exact / checks, 4),
        "verdict_monotonic_datasets": f"{monotone}/{n}",
        "note": "Scenario and baseline share random draws (common random numbers), so the end-balance delta "
        "equals the purchase exactly and bigger purchases never get a milder verdict.",
    }


# ============================================================================ 5. chat intent routing
CHAT_CASES = [
    ("Can I afford a trip next month?", ["estimate_trip_cost", "simulate"]),
    ("can i afford a 45k laptop in december", ["simulate"]),
    ("Should I buy a ₹1.2 lakh bike?", ["simulate"]),
    ("what if i spend 30000 on a phone next week", ["simulate"]),
    ("How much did I spend on food delivery last month?", ["get_spending"]),
    ("how much have I spent on groceries this month", ["get_spending"]),
    ("what's my dining spend in the last 90 days?", ["get_spending"]),
    ("how much did I spend at Starbucks?", ["get_spending"]),
    ("what are my subscriptions", ["get_subscriptions"]),
    ("list subscriptions please", ["get_subscriptions"]),
    ("Am I on track for my emergency fund?", ["get_goals"]),
    ("how are my goals doing", ["get_goals"]),
    ("what will my balance be in 60 days", ["get_forecast"]),
    ("where is my balance heading?", ["get_forecast"]),
    ("how can I save money?", ["get_recommendations"]),
    ("any advice to cut costs", ["get_recommendations"]),
    ("anything unusual in my transactions?", ["get_insights"]),
    ("did I get charged twice somewhere? duplicate?", ["get_insights"]),
    ("hello", ["get_overview"]),
    ("give me a summary", ["get_overview"]),
]


def eval_chat_routing() -> dict[str, Any]:
    import os
    import tempfile

    from finpilot.agents.chat import ChatAgent
    from finpilot.agents.coordinator import Coordinator
    from finpilot.db import configure_database, init_db, session_scope
    from finpilot.llm.client import set_router
    from finpilot.services.seed import seed_demo
    from finpilot.services.toolkit import FinanceToolkit

    tmp = tempfile.mkdtemp()
    configure_database(f"sqlite:///{os.path.join(tmp, 'eval.db')}")
    init_db()
    set_router(OFFLINE)
    with session_scope() as s:
        seed_demo(s, end=END)
    Coordinator(router=OFFLINE, use_jev=False).run_pipeline(trigger="manual")
    hits, rows = 0, []
    with session_scope() as s:
        kit = FinanceToolkit(s)
        agent = ChatAgent(router=OFFLINE, use_jev=False)
        for q, expected in CHAT_CASES:
            tools = [t["tool"] for t in agent.answer(kit, q)["trace"]]
            ok = tools == expected
            hits += ok
            rows.append({"question": q, "expected": expected, "got": tools, "ok": ok})
    return {
        "cases": len(CHAT_CASES),
        "accuracy": round(hits / len(CHAT_CASES), 4),
        "failures": [r for r in rows if not r["ok"]],
        "mode": "deterministic intent router (LLM-free fallback)",
    }


# ============================================================================ report
def render_markdown(r: dict[str, Any]) -> str:
    c = r["categorization"]
    lines = [
        f"# FinPilot self-evaluation ({r['generated_at']})",
        "",
        f"Mode: **{r['mode']}** · Test data: {c['transactions']:,} synthetic transactions "
        f"({c['datasets']} statements, 3 personas × {len(r['seeds'])} seeds, "
        f"{c['long_tail_share'] * 100:.1f}% from long-tail merchants absent from our lexicon).",
        "",
        "## 1. Categorization",
        "",
        "| Pipeline stage | Accuracy | Macro-F1 | Known merchants | Long-tail merchants |",
        "|---|---|---|---|---|",
    ]
    for mode, m in c["ablation"].items():
        lt = f"{m['accuracy_long_tail'] * 100:.1f}%" if m["accuracy_long_tail"] is not None else "–"
        lines.append(
            f"| {mode} | {m['accuracy'] * 100:.1f}% | {m['macro_f1']:.3f} | "
            f"{m['accuracy_known_merchants'] * 100:.1f}% | {lt} |"
        )
    lines += [
        f"| baseline: always '{c['baseline_majority_class']['predicts']}' | "
        f"{c['baseline_majority_class']['accuracy'] * 100:.1f}% | – | – | – |",
        "",
    ]
    d = r["detectors"]
    lines += [
        "## 2. Pattern & anomaly detection",
        "",
        "| Detector | Precision | Recall | F1 |",
        "|---|---|---|---|",
        f"| Recurring series (subscriptions, bills, salary) | {d['recurring_series']['precision']:.3f} | "
        f"{d['recurring_series']['recall']:.3f} | {d['recurring_series']['f1']:.3f} |",
        f"| Transaction anomalies (injected) | {d['anomalies']['precision']:.3f} | {d['anomalies']['recall']:.3f} "
        f"| {d['anomalies']['f1']:.3f} |",
        "",
        "Anomaly recall by type: "
        + ", ".join(f"{k} {v * 100:.0f}%" for k, v in d["anomalies"]["recall_by_type"].items())
        + ".",
        f"Unused-membership detection (Zomato Gold with no Zomato orders): "
        f"{d['unused_membership_detection']['exact_match']}/{d['unused_membership_detection']['datasets']} "
        f"datasets exactly right.",
        "",
    ]
    f = r["forecast"]
    lines += [
        "## 3. Forecast backtest (rolling origin)",
        "",
        f"{f['origins']} forecast origins. Error = {f['metric']} (lower is better).",
        "",
        "| Horizon | FinPilot (P50) mean / median | Naive: last month repeats, mean / median | "
        "Mean of last 3 months | P10–P90 coverage |",
        "|---|---|---|---|---|",
    ]
    for h, row in f["horizons"].items():
        lines.append(
            f"| {h} days | **{row['finpilot_p50']:.1f}%** / {row['finpilot_p50_median']:.1f}% | "
            f"{row['naive_last_30d']:.1f}% / {row['naive_last_30d_median']:.1f}% | "
            f"{row['mean_last_90d']:.1f}% | {row['p10_p90_coverage'] * 100:.0f}% (n={row['n']}) |"
        )
    lines += [
        "",
        "30-day error by persona: "
        + ", ".join(f"{k} {v:.1f}%" for k, v in f["error_30d_by_persona"].items())
        + " (the freelancer's irregular income is the hardest case).",
    ]
    o = f["category_wape_30d_overall"]
    lines += [
        "",
        f"Category-level 30-day spend WAPE: FinPilot **{o['finpilot'] * 100:.1f}%** vs naive "
        f"{o['naive_last_30d'] * 100:.1f}%.",
        "",
    ]
    s = r["simulation"]
    lines += [
        "## 4. What-if simulation consistency",
        "",
        f"{s['scenarios']} purchase scenarios: exact-delta rate {s['exact_delta_rate'] * 100:.0f}%, verdict "
        f"monotonic in {s['verdict_monotonic_datasets']} datasets. {s['note']}",
        "",
    ]
    ch = r["chat"]
    lines += [
        "## 5. Conversational routing",
        "",
        f"{ch['cases']} labelled questions → correct tool sequence in **{ch['accuracy'] * 100:.0f}%** "
        f"({ch['mode']}).",
        "",
    ]
    if r.get("jev"):
        from evaluation import jev_eval

        lines += jev_eval.render(r["jev"])
    if r.get("llm_note"):
        lines += ["## LLM layer", "", r["llm_note"], ""]
    lines += [
        "## Caveats (read before quoting these numbers)",
        "",
        "- All data is synthetic. Known-merchant accuracy is measured against labels drawn from the same merchant "
        "knowledge the categorizer uses, so it mainly tests robustness to noisy bank descriptions (UPI refs, POS "
        "masks, order codes). The **long-tail column is the honest generalisation test**.",
        "- Injected anomalies (₹55k late-night electronics, duplicate charges, fees, a large first-time payee) are "
        "realistic but clear-cut; subtle fraud is out of scope.",
        "- Forecast coverage below the nominal 80% means the P10–P90 band is still somewhat too narrow, mostly on "
        "the freelancer persona whose income is irregular.",
        "- Category-level 30-day spend is inherently noisy (a handful of lumpy purchases per month); both FinPilot "
        "and the naive baseline have high WAPE there.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate the LLM categorization layer")
    ap.add_argument(
        "--jev", action="store_true", help="measure System One (Jev) decisions live and record them"
    )
    ap.add_argument(
        "--jev-replay", action="store_true", help="measure Jev from the recorded cassette (offline)"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    llm_router = None
    note = (
        "Deterministic run: no LLM calls were made. Run `python -m evaluation.run_eval --llm` with "
        "MESH_API_KEY set to measure the LLM long-tail stage (row 4) on the same data."
    )
    if args.llm:
        llm_router = get_router()
        if not llm_router.available("categorizer"):
            print("LLM requested but no provider configured — set MESH_API_KEY", file=sys.stderr)
            sys.exit(2)
        note = f"LLM stage measured with {llm_router.describe('categorizer')}."
    t0 = time.perf_counter()
    dsets = datasets(args.seeds)
    jev_client = None
    if args.jev or args.jev_replay:
        import tempfile

        from evaluation.jev_eval import make_client
        from finpilot.db import configure_database, init_db

        # decision logs and call telemetry from the evaluation go to a throwaway database
        configure_database(f"sqlite:///{tempfile.mkdtemp()}/jev-eval.db")
        init_db()
        jev_client = make_client("replay" if args.jev_replay else "live")
        if args.jev_replay and not jev_client.tape:
            jev_client = None  # nothing recorded yet: the Jev rows are reported as pending
        elif args.jev and not jev_client.available():
            print("--jev needs MESH_API_KEY (or JEV_PROVIDER=typesafe + TYPESAFE_API_KEY)", file=sys.stderr)
            sys.exit(2)
        elif args.jev:
            from evaluation.jev_eval import JevEvalError, preflight

            try:
                print(f"jev preflight … {preflight(jev_client)}", flush=True)
            except JevEvalError as exc:
                print(f"\n{exc}", file=sys.stderr)
                sys.exit(2)
    print("categorization …", flush=True)
    cat = eval_categorization(dsets, llm_router, jev_client)
    if jev_client is not None:
        jev_client.save()
    print("detectors …", flush=True)
    det = eval_detectors(dsets)
    print("forecast backtest …", flush=True)
    fc = eval_forecast(args.seeds[:2])
    print("simulation …", flush=True)
    sim = eval_simulation(args.seeds[:2])
    print("chat routing …", flush=True)
    chat = eval_chat_routing()
    print("system one (jev) …", flush=True)
    from evaluation import jev_eval

    jev = jev_eval.run("live" if args.jev else ("replay" if args.jev_replay else None))
    result = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": "rules+LLM" if args.llm else "deterministic (no LLM)",
        "seeds": args.seeds,
        "categorization": cat,
        "detectors": det,
        "forecast": fc,
        "simulation": sim,
        "chat": chat,
        "jev": jev,
        "llm_note": note,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(result, indent=2, default=str))
    (OUT / "REPORT.md").write_text(render_markdown(result))
    print(render_markdown(result))


if __name__ == "__main__":
    main()
