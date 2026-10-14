# API agents and financial evidence

The active workflow uses LLM APIs: Mesh for submission and FreeLLMAPI for development. Jev is disabled
by default and is not called by the default agents; its optional code and historical tests remain.

## Responsibilities

- **Coordinator:** LangGraph runs categorize → analyze → forecast → advise → audit, with scheduled digests.
  The plan step records *why* each step runs, the transactions needing review and the model each agent
  uses. Missing balance provenance no longer skips advice. Critical failures stop dependent work. Changed
  transaction, balance or goal inputs invalidate stale results. Steps record model calls and decisions.
- **Categorizer:** memory → curated lexicon (family rules keep the matched brand, e.g. "Third Wave Coffee",
  not "Café") → keyword *phrases* (a matched phrase hides the words inside it: "MILK PARLOUR" is a dairy) →
  person-to-person detection (needs a common Indian name or a phone-number VPA; bank "P2A" markers count) →
  **one LLM decision per uncertain merchant + direction** (79 groups instead of ~600 narrations on the demo)
  → merchant consistency. The LLM prompt separates payee business from payment purpose, uses brand knowledge
  (with examples that are deliberately *not* in the evaluation hold-out) and calibrated confidence. An unsure
  model never erases specific evidence; a curated rule yields only to a confident model. P2P payments are
  `transfers` with review-level confidence because the purpose is unknown. Manual/import labels and user
  memory stay authoritative. Every transaction also records **direction** and **payment channel**.
- **Analyzer:** Exact aggregates establish trends (with the merchants driving each spike), month-to-date pace
  against the same days of last month, anomalies, debt obligations, bank cash flow and the review queue.
  Its prompt writes 3-4 ordered observations, copying figures exactly. Bank history alone cannot prove a
  membership unused; that claim requires explicit usage confirmation.
- **Forecaster:** Recurring obligations plus historical variable cash flow produce seeded simulation
  paths. Shared masked loan references link narration changes. Monthly card obligations include every
  recorded bill across platforms, without asserting one card account. Varying bills/income sample actual
  observed amounts. Scenario paths share draws; invalid amounts, cuts below -100% and out-of-horizon dates
  are rejected. The narrative prompt copies figures exactly (strict grounding rejects rounding).
- **Advisor:** An API decision step reasons over patterns, obligations, trajectory and goals to prioritize
  evidence-derived actions; a separate phrasing step explains them (evidence → action → effect). New actions:
  confirm the review queue, slow a month-to-date spending pace. Every recommendation stores its trigger,
  transaction/forecast evidence (including derived figures such as a funding plan's share of a goal gap)
  and reasoning.
- **Auditor:** Deterministic and independent of the models it checks. Each recommendation's trigger must
  match a finding or forecast of the same run, evidence transaction ids must exist in the ledger, the trail
  needs at least two reasoning steps and a finite impact, and every figure in the title/message must be
  grounded in its evidence. The verdict is appended to the reasoning; failures are marked ungrounded. It also
  reports categorization health (share of outflows needing review).
- **Chat team:** guardrails (actions, injection, off-topic) → retriever (value linking to the user's merchants
  and categories, BM25 over insights/recommendations/goals) → SQL Analyst for data questions (deterministic
  parser when every word is explained, otherwise LLM text-to-SQL over a sandboxed per-user SQLite view with
  schema, date anchors and retrieved values, repaired from database errors) or the planner for finance tools
  → composer + verifier. Answers carry exact table/chart artifacts. "Biggest expense" means an individual
  expense debit; payee matching uses whole names (CRED totals exclude Kiwi mentioning ICICI Cred); relative
  periods use the current calendar; history resolves references but never establishes facts.
- **Digest:** Its own prompt summarizes one supported insight and action. APScheduler drives delivery;
  an empty recommendation list does not mean all steady. Local verification never sends external messages.

## Unknown balance

Without a verified starting balance, forecast bands explicitly show **net cash-flow changes**, computed
from zero as an offset, not account balances. Starting balance and overdraft probabilities are null;
goal odds are withheld. The full pipeline still produces transaction-grounded advice. Simulations return
`needs_balance` with the exact incremental effect rather than yes/no affordability. Required starting
cash is a conditional model estimate. The UI labels charts and results accordingly.

## Claim checks and gateway errors

Strict numeric grounding and an independent API fact checker validate identities, scope, period,
causes and missing-data assumptions. Duplicate verdict IDs and string booleans are invalid. Unsupported
generated text falls back to verified calculations/templates and records why. A model fact checker is
not a proof of correctness. Incomplete JSON is never cached; truncated structured responses get one
bounded retry with more output capacity.

FreeLLMAPI can advertise a large context window while a route's configured token-per-minute limit is too
low for financial prompts. The local Qwen route was limited to 1,000 TPM and returned HTTP 413 as a
context/token error. Development uses `auto:smart` for sufficient-capacity routing. Actual models and
failures remain in the model-call trail.

For submission, select **Mesh** and an available Mesh model in Settings, clear development per-agent
overrides and provide `MESH_API_KEY`. Fresh databases default to Mesh. Persisted Settings overrides take
precedence over `.env`; changing environment variables alone does not replace a saved development choice.

## Measurement

`tests/test_api_agents.py` covers zero-advice regressions, ranking, repayment identities, varying bills,
exact deltas, unknown-balance claims, usage evidence and invalid verdicts. Existing isolation, agent,
API, ingestion, scheduler and optional Jev tests also run on SQLite and a separate PostgreSQL database.

`python -m evaluation.run_eval --seeds 1 2 3` records reproducible synthetic categorization (with a frozen
long-tail hold-out and precision when confident), a hand-labelled real-narration benchmark, detector,
forecast, simulation and end-to-end chat answer accuracy in `evaluation/results/latest.json` and `REPORT.md`. Offline results
do not measure live LLM accuracy. Live API cases are reported separately. Forecast coverage remains
below the nominal 80%, particularly for irregular freelance income; accuracy is not guaranteed.
