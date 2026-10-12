# FinPilot self-evaluation (2026-10-12 10:18)

Mode: **deterministic (no LLM)** · Test data: 5,613 synthetic transactions (9 statements, 3 personas × 3 seeds, 21.8% from long-tail merchants absent from our lexicon).

## 1. Categorization

| Pipeline stage | Accuracy | Macro-F1 | Known merchants | Long-tail merchants |
|---|---|---|---|---|
| 1_lexicon_only | 79.0% | 0.825 | 100.0% | 3.8% |
| 2_plus_keywords | 93.3% | 0.901 | 100.0% | 69.4% |
| 3_plus_patterns (deterministic) | 93.3% | 0.901 | 100.0% | 69.4% |
| baseline: always 'food_delivery' | 19.8% | – | – | – |

## 2. Pattern & anomaly detection

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Recurring series (subscriptions, bills, salary) | 1.000 | 0.973 | 0.986 |
| Transaction anomalies (injected) | 0.973 | 1.000 | 0.986 |

Anomaly recall by type: large_new_merchant 100%, duplicate_charge 100%, large_purchase 100%, fee 100%.
Unused-membership detection (Zomato Gold with no Zomato orders): 3/3 datasets exactly right.

## 3. Forecast backtest (rolling origin)

42 forecast origins. Error = mean |predicted − actual net cash-flow| as % of monthly income (lower is better).

| Horizon | FinPilot (P50) mean / median | Naive: last month repeats, mean / median | Mean of last 3 months | P10–P90 coverage |
|---|---|---|---|---|
| 30 days | **28.4%** / 13.3% | 66.1% / 63.2% | 55.3% | 71% (n=42) |
| 60 days | **41.0%** / 18.3% | 104.5% / 73.7% | 88.9% | 75% (n=36) |
| 90 days | **51.9%** / 22.5% | 109.1% / 76.1% | 95.9% | 73% (n=30) |

30-day error by persona: young_professional 14.1%, family 13.5%, freelancer 57.7% (the freelancer's irregular income is the hardest case).

Category-level 30-day spend WAPE: FinPilot **52.3%** vs naive 57.4%.

## 4. What-if simulation consistency

36 purchase scenarios: exact-delta rate 100%, verdict monotonic in 6/6 datasets. Scenario and baseline share random draws (common random numbers), so the end-balance delta equals the purchase exactly and bigger purchases never get a milder verdict.

## 5. Conversational routing

20 labelled questions → correct tool sequence in **100%** (deterministic intent router (LLM-free fallback)).

## LLM layer

Deterministic run: no LLM calls were made. Run `python -m evaluation.run_eval --llm` with MESH_API_KEY set to measure the LLM long-tail stage (row 4) on the same data.
