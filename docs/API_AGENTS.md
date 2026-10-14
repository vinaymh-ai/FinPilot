# API agents and financial evidence

The active workflow uses LLM APIs: Mesh for submission and FreeLLMAPI for development. Jev is disabled
by default and is not called by the default agents; its optional code and historical tests remain.

## Responsibilities

- **Coordinator:** LangGraph runs categorize → analyze → forecast → advise, with scheduled digests.
  Missing balance provenance no longer skips advice. Critical failures stop dependent work. Changed
  transaction, balance or goal inputs invalidate stale results. Steps record model calls and decisions.
- **Categorizer:** API reasoning identifies the payee's business separately from transaction purpose,
  using full narration, signed amount, date, currency and fallible rule suggestions. Uncertain intermediary
  payments receive a second reasoning pass. Company words such as Club do not prove a membership fee.
  Explicit rent/order/card evidence wins over unsupported inference. Manual/import labels and user memory
  stay authoritative. Invalid model IDs, taxonomy values or confidence values are rejected. Unresolved
  ambiguity requires review; model confidence is not a measured probability of correctness.
- **Analyzer:** Exact aggregates establish trends, anomalies, debt obligations and bank cash flow.
  Its own prompt interprets those facts with coverage caveats. Transfers/investments can leave the bank
  without counting as consumption spending. Card bills do not reveal original purchases. Bank history
  alone cannot prove a membership unused; that claim requires explicit usage confirmation.
- **Forecaster:** Recurring obligations plus historical variable cash flow produce seeded simulation
  paths. Shared masked loan references link narration changes. Monthly card obligations include every
  recorded bill across platforms, without asserting one card account. Varying bills/income sample actual
  observed amounts. Weekday Bayesian bootstrap weights model uncertainty in estimated historical rates.
  Scenario paths share draws; invalid amounts, cuts below -100% and out-of-horizon dates are rejected.
- **Advisor:** An API decision step reasons over patterns, obligations, trajectory and goals to prioritize
  evidence-derived actions. A separate phrasing step explains them. Every recommendation stores its trigger,
  transaction/forecast evidence and reasoning. Review actions invent no savings or loan-interest benefit.
- **Chat:** API query planning selects read-only tools; code calculates exact results; API generation
  explains them. “Biggest expense” means an individual expense debit, distinct from merchant/category
  totals. Matching uses payee identity rather than arbitrary bank/creditor text: CRED totals exclude Kiwi
  merely mentioning ICICI Cred. Relative periods use the current calendar with explicit boundaries.
  Chat history resolves references but never establishes financial facts. Greetings need no ledger access.
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

`python -m evaluation.run_eval --seeds 1 2` records reproducible synthetic categorization, detector,
forecast and simulation metrics in `evaluation/results/latest.json` and `REPORT.md`. Offline results
do not measure live LLM accuracy. Live API cases are reported separately. Forecast coverage remains
below the nominal 80%, particularly for irregular freelance income; accuracy is not guaranteed.
