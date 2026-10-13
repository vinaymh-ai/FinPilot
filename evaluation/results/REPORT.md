# FinPilot self-evaluation (2026-10-12 21:29)

Mode: **deterministic (no LLM)** · Test data: 5,631 synthetic transactions (9 statements, 3 personas × 3 seeds, 21.8% from long-tail merchants absent from our lexicon).

## 1. Categorization

| Pipeline stage | Accuracy | Macro-F1 | Known merchants | Long-tail merchants |
|---|---|---|---|---|
| 1_lexicon_only | 79.0% | 0.827 | 100.0% | 3.8% |
| 2_plus_keywords | 93.4% | 0.903 | 100.0% | 69.7% |
| 3_plus_patterns (deterministic) | 93.4% | 0.903 | 100.0% | 69.7% |
| baseline: always 'food_delivery' | 20.0% | – | – | – |

## 2. Pattern & anomaly detection

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Recurring series (subscriptions, bills, salary) | 1.000 | 0.973 | 0.986 |
| Transaction anomalies (injected) | 0.973 | 1.000 | 0.986 |

Anomaly recall by type: fee 100%, duplicate_charge 100%, large_new_merchant 100%, large_purchase 100%.
Unused-membership detection (Zomato Gold with no Zomato orders): 3/3 datasets exactly right.

## 3. Forecast backtest (rolling origin)

42 forecast origins. Error = mean |predicted − actual net cash-flow| as % of monthly income (lower is better).

| Horizon | FinPilot (P50) mean / median | Naive: last month repeats, mean / median | Mean of last 3 months | P10–P90 coverage |
|---|---|---|---|---|
| 30 days | **28.9%** / 14.7% | 65.6% / 63.2% | 54.7% | 67% (n=42) |
| 60 days | **41.6%** / 19.2% | 103.5% / 73.7% | 88.0% | 72% (n=36) |
| 90 days | **53.1%** / 28.4% | 107.5% / 76.1% | 94.3% | 63% (n=30) |

30-day error by persona: young_professional 15.5%, family 13.5%, freelancer 57.7% (the freelancer's irregular income is the hardest case).

Category-level 30-day spend WAPE: FinPilot **52.2%** vs naive 57.3%.

## 4. What-if simulation consistency

36 purchase scenarios: exact-delta rate 100%, verdict monotonic in 6/6 datasets. Scenario and baseline share random draws (common random numbers), so the end-balance delta equals the purchase exactly and bigger purchases never get a milder verdict.

## 5. Conversational routing

20 labelled questions → correct tool sequence in **100%** (deterministic intent router (LLM-free fallback)).

## 6. System One decisions (Jev)

Model `typesafe/jev-1.13` via Mesh `POST /v1/evaluate`. Status: **pending — no Jev measurements in this run**.

### Deterministic paths Jev replaces or augments (measured)

| Decision | Deterministic path | Result |
|---|---|---|
| Merchant category (142 hand-labelled merchants) | lexicon + keywords | 68.3% overall; **26.9%** on the 52 it is unsure about (these go to Jev) |
| Chat routing (62 messages) | regex intent router | 50.0% intent accuracy; cannot refuse the 12 unsafe/off-topic messages or spot the 6 that need reasoning |
| Faithfulness (24 drafts, 18 unfaithful) | numeric grounding check | catches **22%** of unfaithful drafts — 14 contain no wrong number at all |
| Goal type (24 goal names) | keyword table | type 75%, spend-vs-keep 92% |
| CSV columns (20 headers) | header vocabulary | 2 headers unrecognised |

### Cost per 1,000 decisions (request sizes measured from the real request bodies; ≈4 chars/token)

| Decision | Input tokens / request | Jev $ / 1k | LLM it replaces, $ / 1k (est.) |
|---|---|---|---|
| merchant_category | ~1004 | $0.0422 | $0.0396 |
| chat_route | ~1202 | $0.0505 | $0.6675 |
| faithfulness | ~230 | $0.0097 | — (new check) |
| goal_type | ~340 | $0.0143 | — (new check) |
| csv_column | ~219 | $0.0092 | — (new check) |

Jev accuracy, calibration and thresholds: **pending** — MESH_API_KEY=… python -m evaluation.run_eval --jev   (records evaluation/cassettes/jev.json; afterwards --jev-replay reproduces every number offline).

## LLM layer

Deterministic run: no LLM calls were made. Run `python -m evaluation.run_eval --llm` with MESH_API_KEY set to measure the LLM long-tail stage (row 4) on the same data.

## Caveats (read before quoting these numbers)

- All data is synthetic. Known-merchant accuracy is measured against labels drawn from the same merchant knowledge the categorizer uses, so it mainly tests robustness to noisy bank descriptions (UPI refs, POS masks, order codes). The **long-tail column is the honest generalisation test**.
- Injected anomalies (₹55k late-night electronics, duplicate charges, fees, a large first-time payee) are realistic but clear-cut; subtle fraud is out of scope.
- Forecast coverage below the nominal 80% means the P10–P90 band is still somewhat too narrow, mostly on the freelancer persona whose income is irregular.
- Category-level 30-day spend is inherently noisy (a handful of lumpy purchases per month); both FinPilot and the naive baseline have high WAPE there.
