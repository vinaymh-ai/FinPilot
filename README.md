# FinPilot — a proactive, multi-agent financial co-pilot

> Most finance apps tell you what already happened. FinPilot works out where your money is **heading**,
> answers *"can I afford this?"* **before** you commit, and tells you **one specific thing to do** — with the
> evidence behind it.

FinPilot ingests a bank statement, and a team of cooperating agents categorises every transaction, studies
behaviour over time, simulates 400 possible futures from your own habits, and turns all of it into ranked,
quantified, *auditable* recommendations. A background scheduler delivers a weekly digest without being asked.

**Every LLM call routes through [Mesh API](https://developers.meshapi.ai)** (`https://api.meshapi.ai/v1`) by
default — and the system still runs end-to-end with **zero LLM spend** (each agent has a deterministic path).

![Overview](docs/screenshots/overview.png)

| What-if Lab | Decision trail ("Why?") |
|---|---|
| ![What-if](docs/screenshots/what-if-lab.png) | ![Decision trail](docs/screenshots/decision-trail.png) |
| **Agents & run trail** | **Ask FinPilot (tool-calling chat)** |
| ![Agents](docs/screenshots/agents.png) | ![Chat](docs/screenshots/chat.png) |

---

## Contents
1. [What it does — the story](#1-what-it-does--the-story)
2. [Quick start](#2-quick-start)
3. [How the agents cooperate](#3-how-the-agents-cooperate)
4. [Forecasting & what-if simulation](#4-forecasting--what-if-simulation)
5. [Auditable decision trail](#5-auditable-decision-trail)
6. [Self-evaluation results](#6-self-evaluation-results)
7. [Key design decisions](#7-key-design-decisions)
8. [LLM providers — Mesh by default, bring your own](#8-llm-providers--mesh-by-default-bring-your-own)
9. [API, project layout, deployment](#9-api-project-layout-deployment)
10. [Limitations & roadmap](#10-limitations--roadmap)

---

## 1. What it does — the story

The demo account ("Arjun", Bengaluru, 8 months of messy UPI/POS/NEFT statement lines) loads on first boot.
After one agent run, FinPilot says — from Arjun's own data (figures from a run on 12 Oct 2026; the demo is
generated relative to *today*, so exact numbers shift):

| Capability | What FinPilot surfaces |
|---|---|
| **Categorisation** | 609 transactions labelled with *how* (merchant lexicon / keyword evidence / LLM / your correction / recurring pattern), a confidence and a reason. Correct one Truffles row → every Truffles row (and future imports) follow. |
| **Patterns & anomalies** | "Food delivery is up 47% in the last 30 days (₹12,802 vs ₹8,694, 26 orders)" · "3 overlapping video-streaming services" · "**Zomato Gold looks unused — no Zomato orders in 93 days**" · "Netflix price +23%" · "Possible duplicate Swiggy charge" · "₹54,990 late-night electronics purchase is 46× your typical shopping ticket" |
| **Forecast** | P10/P50/P90 balance for 6 months, the tightest day, overdraft risk, and per-goal probability: *"Emergency fund is 2% likely by 31 Mar 2027 — ₹73k short at this pace; you'd get there around May 2027."* |
| **What-if** | *"Can I afford a ₹45k trip next month?"* → **A stretch**: pushes the emergency fund back ~6 weeks; *"cut food delivery ₹10.2k→₹5.2k, dining ₹3.4k→₹1.7k … for 6 months to fund it"* (or *"wait until <date>"* when that works). |
| **Advice** | Ranked, quantified actions: cancel Zomato Gold (₹349/mo), keep Netflix & drop Prime Video + Hotstar (₹598/mo), cap food delivery at your usual ₹8.7k (₹4.1k/mo), dispute the duplicate charge, autopay to stop late fees — and an emergency-fund plan built from those levers. |
| **Proactive** | APScheduler cron → weekly digest (in-app inbox, email via SMTP, Telegram bot) with **one key insight + one action**. |
| **Conversational** | "How much did I spend on food delivery last month?" · "Am I on track?" · "Can I afford a ₹1.2L bike in December?" — answered through tool calls over real data, with the tool trace shown. |

---

## 2. Quick start

**Prerequisites:** Python 3.11+, Node 20+ (only to build the UI).

```bash
git clone https://github.com/vinaymh-ai/FinPilot && cd FinPilot
cp .env.example .env              # add MESH_API_KEY=rsk_...  (optional — see "zero-cost mode")

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
(cd frontend && npm ci && npm run build)

uvicorn finpilot.main:app --port 8000   # → http://localhost:8000  (UI + API + /docs)
```

On first boot the demo account is seeded and the full agent graph runs once. Upload your own CSV on
**Transactions** or switch demo personas in **Settings** (young professional · family with EMIs · freelancer with
irregular income).

**Frontend dev mode:** `uvicorn finpilot.main:app --reload` + `cd frontend && npm run dev` → http://localhost:5173
(Vite proxies `/api`). **Docker:** `docker compose up --build`. **Make targets:** `make setup | api | web | test | eval`.

**Zero-cost mode.** Without a key (or with `LLM_ENABLED=false`) every agent completes its job deterministically —
93% categorisation accuracy, full forecasting, simulation, advice with template text, and a rule-based chat that
calls the same tools. Adding a Mesh key upgrades the long-tail categorisation, narratives, advice phrasing and the
free-form chat. You can also point it at a free provider (Groq, Gemini free tier, OpenRouter `:free`, local Ollama).

---

## 3. How the agents cooperate

A **LangGraph** `StateGraph` (the *Coordinator*) plans each run and routes work between specialists. Every node is
wrapped so it records an `AgentStep` (timing, summary, structured details, LLM calls, errors) under one `AgentRun`.

```mermaid
graph LR
    S([start]) --> plan
    plan -- new / uncategorised txns --> categorize
    plan -- nothing new --> analyze
    plan -- no data --> E([end])
    categorize --> analyze
    analyze -- critical failure --> E
    analyze --> forecast --> advise
    advise -- scheduled run --> digest --> E
    advise --> E
```

| Agent | Its job | Own prompt / tools | Deterministic path | Where the LLM adds value |
|---|---|---|---|---|
| **Coordinator** | Plans the run (skip categorisation if nothing is new, add digest on scheduled runs), routes, records the trail | LangGraph conditional edges | always | — |
| **Categorizer** | Label every transaction with category + confidence + reason | Categorisation prompt with full taxonomy & Indian-retail hints | memory → history → ~200-brand lexicon → keyword evidence (amount- & sign-aware, P2P detection) → recurring-pattern post-pass | Long-tail local merchants, **batched once per unique merchant** (300 Swiggy rows never reach the LLM) |
| **Analyzer** | Behaviour over time; patterns & anomalies | Behaviour-profile prompt | rolling 30-day windows vs previous three (z-scores), median/MAD outliers, duplicate/fee/new-payee detection, recurring-series detection, subscription overlap & **usage evidence**, impulse tagging | 3–4 bullet behaviour profile (grounding-checked) |
| **Forecaster** | Project the trajectory; simulate decisions | Forecast-narrative prompt | Monte-Carlo engine (below), goal funding, what-if, "wait until"/"cut" suggestions | Plain-language explanation (grounding-checked) |
| **Advisor** | Turn signals into specific, ranked actions | Advisor prompt (rewrite, never invent numbers) | 11 candidate generators, priority scoring, feedback adaptation, goal-progress adaptation | Friendlier phrasing + weekly headline, rejected if any figure isn't in the evidence |
| **Chat** | Free-form money questions | System prompt + **10 function-calling tools** (`simulate`, `get_spending`, `get_forecast`, …) | Intent router that calls the *same* tools | Open-ended questions, multi-step tool use |
| **Digest** | Weekly proactive message: one insight + one action | Digest prompt | Template composer | Warm, specific opener (grounding-checked) |

**One run, step by step (demo account):**
1. `plan` — 609 transactions, 609 uncategorised → plan `categorize → analyze → forecast → advise`.
2. `categorize` — 502 by lexicon, 89 by keyword evidence, 18 uncertain → batched to the LLM if configured.
3. `analyze` — 14 recurring series, 8 subscriptions, 4 anomalies, 12 findings → persisted as insights; anomalies
   and impulse tags written back to transactions.
4. `forecast` — 400 paths × 185 days in ~50 ms; goal snapshots appended (used next run to adapt advice).
5. `advise` — 9 candidates → 8 ranked recommendations, each with trigger + evidence + reasoning.
6. `digest` (scheduled runs) — composed and delivered; per-channel delivery status stored.

---

## 4. Forecasting & what-if simulation

*"If you keep behaving like this, here is what happens."*

* **Known commitments** — salary, rent, EMIs, bills and subscriptions are scheduled on their own cadence. Monthly
  series use a **slot model** (anchor day-of-month), so a salary paid early on the 27th because the 1st is a Sunday
  counts for *next* month instead of being double-counted.
* **Everything else** — daily variable cash-flow is **bootstrapped from your last 90 days**, stratified by weekday
  (weekends look like weekends) and recency-weighted (the last 30 days count double, so a new habit shows up).
* **One-off shocks** — anomalies are removed from the bootstrap (a ₹55k TV isn't a daily habit) but their observed
  **frequency and size** are kept as a compound-Poisson risk term, so the P10 band stays honest.
* **Goals** — funded in priority order from balance above a safety buffer (auto: ½ month of fixed + essential
  costs). Per-path funding gives probability, shortfall, ETA and the monthly top-up that closes the gap.
* **What-if** — one-off purchases, new recurring costs, category cuts, cancelled subscriptions, income changes —
  combinable. Baseline and scenario use **common random numbers** (identical draws), so the reported delta is the
  effect of the decision, not Monte-Carlo noise. Verdicts: *affordable / a stretch / not affordable*, plus
  *"wait until <date>"* (searched week by week) and a funding plan from your discretionary categories.

---

## 5. Auditable decision trail

Every recommendation stores:

* **trigger** — which agent and signal fired it (`analyzer:subscription_unused`, `forecaster:goal_at_risk`, …)
* **evidence** — the exact numbers and transaction ids behind it
* **reasoning** — ordered steps, including the priority formula
  (`impact % of income × confidence × urgency × feedback weight`) and how it was phrased
  (`template` or `mesh:openai/gpt-4o-mini — grounding check passed`)
* the **run id** → the full step timeline of the run that produced it, and the history of the same
  recommendation across runs (fingerprints) with your feedback.

In the UI, every card has a **"Why?"** drawer; the **Agents & Trail** page shows the graph, step timeline, per-call
LLM telemetry (tokens, latency, cache hits) and all-time usage.

---

## 6. Self-evaluation results

`python -m evaluation.run_eval` (deterministic, no API cost; runs in CI). Full report:
[`evaluation/results/REPORT.md`](evaluation/results/REPORT.md) · raw: [`latest.json`](evaluation/results/latest.json)
· also rendered on the **Evaluation** page.

**Test data:** 5,631 synthetic transactions — 3 personas × 3 seeds × 8 months, generated with ground-truth labels
(`finpilot/services/synthetic.py`). Descriptions are deliberately messy (UPI refs, POS card masks, order codes,
city suffixes) and **21.8% of rows come from long-tail local merchants that are not in the lexicon**
("TRUFFLES", "MTR 1924", "GREEN TRENDS", "SHRI GANESH MEDICALS"…). Anomalies and behaviour shifts are injected.

**Categorisation (ablation):**

| Pipeline stage | Accuracy | Macro-F1 | Long-tail merchants |
|---|---|---|---|
| Baseline: always the most common category | 20.0% | – | – |
| Merchant lexicon only | 79.0% | 0.827 | 3.8% |
| + keyword evidence, P2P & amount-aware rules | **93.4%** | **0.903** | **69.7%** |
| + LLM for uncertain merchants (`--llm`) | run with `MESH_API_KEY` | | |

**Detectors:**

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Recurring series (subscriptions, bills, salary) | 1.000 | 0.973 | 0.986 |
| Transaction anomalies (injected) | 0.973 | 1.000 | 0.986 |
| Unused membership (Zomato Gold, no Zomato orders) | 3/3 datasets exactly right | | |

**Forecast backtest** — 42 rolling month-end origins over 12-month histories; error = |predicted − actual net
cash-flow| as % of monthly income:

| Horizon | FinPilot P50 (mean / median) | Naive: last month repeats (mean / median) | Mean of last 3 months | P10–P90 coverage |
|---|---|---|---|---|
| 30 days | **28.9% / 14.7%** | 65.6% / 63.2% | 54.7% | 67% |
| 60 days | **41.6% / 19.2%** | 103.5% / 73.7% | 88.0% | 72% |
| 90 days | **53.1% / 28.4%** | 107.5% / 76.1% | 94.3% | 63% |

30-day error by persona: family 13.5%, young professional 15.5%, **freelancer 57.7%** (irregular income is the hard
case). **What-if consistency:** 36/36 scenarios have an exact purchase delta; verdicts are monotonic in purchase
size on 6/6 datasets. **Chat routing:** 20/20 labelled questions call the right tool sequence (LLM-free router).

**Caveats, honestly:** data is synthetic; accuracy on *known* merchants (100%) mostly tests normalisation of
noisy descriptions because the labels share the lexicon's knowledge — the long-tail column is the real
generalisation test. Injected anomalies are clear-cut. Coverage under the nominal 80% means the bands are still a
little narrow (mostly for irregular income). Measuring the LLM stage needs a Mesh key: `python -m
evaluation.run_eval --llm`.

---

## 7. Key design decisions

1. **Deterministic-first, LLM where judgment is needed.** Arithmetic, statistics and scheduling are done in code;
   the LLM handles what code can't (unknown merchants, natural language). This keeps the product correct,
   testable and nearly free — and it keeps working when the API key, budget or network doesn't.
2. **Numbers from code, words from the LLM — enforced.** LLM-written advice, narratives and digests pass a
   numeric grounding check (`finpilot/llm/grounding.py`): any figure not derivable from the evidence (with
   rounding, ×12, /12 tolerance) rejects the text and the template is kept. The rejection is recorded in the trail.
3. **Pay per merchant, not per row.** Uncertain transactions are grouped by merchant and batched (40 merchants per
   call); results are cached in the DB and reused as "history" for future imports.
4. **User feedback is a first-class signal.** Category corrections become merchant memory (including prefix
   matches like "Truffles New Branch"); dismissed advice is snoozed 30 days unless the problem grows ≥25%, and
   repeatedly dismissed kinds are down-weighted; goal advice compares with the previous run ("up from 41% → 63%").
5. **Common random numbers for what-if**, so "can I afford it?" isn't answered by noise.
6. **Usage evidence over guesswork.** "Unused subscription" is only claimed when data proves it (a delivery
   membership with no orders from that merchant in 45+ days) or when the user says so — not from a name.
7. **Behaviour ≠ one-offs.** Spikes are computed on variable spend with anomalies and fixed bills removed, so a TV
   purchase or a bill landing twice in a 30-day window doesn't masquerade as a "habit".
8. **One coordinator, explicit graph.** LangGraph gives a readable analyse → forecast → advise flow with
   conditional routing; the step wrapper gives observability and graceful degradation (non-critical node fails →
   run continues; analysis fails → run aborts with the error recorded).
9. **Production basics:** pydantic-settings config, SQLite by default / PostgreSQL via `DATABASE_URL`, idempotent
   CSV imports (row fingerprints), per-user run locks, optional bearer token, structured JSON logs, Docker image
   with health check and non-root user, CI (ruff, pytest, evaluation, frontend build). Keys live only in env vars —
   never in the database or the UI.

---

## 8. LLM providers — Mesh by default, bring your own

```python
# finpilot/llm/providers.py — every provider speaks the OpenAI protocol, so switching is configuration
client = OpenAI(base_url="https://api.meshapi.ai/v1", api_key=os.getenv("MESH_API_KEY"))
```

| Provider | Env var | Default model | Notes |
|---|---|---|---|
| **Mesh API** (default) | `MESH_API_KEY` | `openai/gpt-4o-mini` | one `rsk_` key → GPT, Claude, Gemini, Llama…; hackathon-mandated |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | free tier |
| Google Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` | free tier |
| OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct:free` | `:free` models |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` | OpenAI-compatible endpoint |
| Together | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | |
| Ollama | `OLLAMA_BASE_URL` | `llama3.1:8b` | local, free, private |
| Custom | `CUSTOM_LLM_BASE_URL` / `_API_KEY` | – | any OpenAI-compatible server |

* **Per-agent routing:** e.g. `CATEGORIZER_MODEL=openai/gpt-4o-mini`, `ADVISOR_MODEL=anthropic/claude-haiku-4.5`,
  or switch provider/model per agent at runtime on **Settings** (persisted; keys stay in `.env`). The Settings page
  lists Mesh's live model catalogue and has a **Test connection** button.
* **Fallback chain:** `LLM_FALLBACK_PROVIDERS=groq,ollama` — tried in order if the primary fails; if all fail,
  agents fall back to their deterministic path. Every call is logged (tokens, latency, cache hit, success).

> For the hackathon submission keep `LLM_PROVIDER=mesh` (the default) so that every LLM call goes through Mesh.

---

## 9. API, project layout, deployment

Interactive docs at **`/docs`**. Highlights:

| Endpoint | Purpose |
|---|---|
| `POST /api/transactions/upload` | CSV import (bank-export tolerant, idempotent) → runs the agents |
| `PATCH /api/transactions/{id}` | correct a category → merchant memory |
| `POST /api/runs` | run the agent graph (`recategorize`, `deliver_digest`) |
| `GET /api/runs/{id}` | step timeline + LLM calls (decision trail) |
| `GET /api/dashboard` · `/insights` · `/forecast` · `/recurring` | views |
| `POST /api/simulate` | what-if with any combination of adjustments |
| `GET/PATCH /api/recommendations[/{id}]` | advice + feedback (accepted / done / dismissed) |
| `GET/POST/PATCH/DELETE /api/goals` | goals with live projections and history |
| `POST /api/chat` | tool-calling assistant (returns the tool trace) |
| `GET /api/digests` · `POST /api/digests/send-now` · `PUT /api/scheduler` | proactive delivery |
| `GET/PUT /api/settings…` · `GET /api/llm/models` · `POST /api/settings/llm/test` | profile & provider routing |
| `GET /api/evaluation` | self-evaluation results |

```
finpilot/
  agents/       coordinator (LangGraph) · categorizer · analyzer · forecaster · advisor · chat · digest
  domain/       taxonomy · merchant lexicon · description normaliser · recurring detection · stats
  llm/          provider registry · router (fallback, cache, telemetry) · grounding check
  services/     ingestion · synthetic data · repository · toolkit (chat tools) · scheduler · notifier · seed
  api/          FastAPI routers + serializers        main.py · models.py · config.py · db.py
evaluation/     run_eval.py + results/ (REPORT.md, latest.json)
frontend/       React 19 + Vite + TypeScript + Tailwind 4 + Recharts + TanStack Query
tests/          40 tests: domain logic, agents (incl. fake-LLM tool calling & grounding), ingestion, API
```

**Deploy for free:** `render.yaml` (Render free web service, Docker), or any Docker host / Hugging Face Docker
Space (`PORT` is honoured). Use a free Postgres (Neon/Supabase) via `DATABASE_URL=postgresql+psycopg://…` for
persistence. Run exactly one instance with `SCHEDULER_ENABLED=true`.

**Hackathon checks:** `.github/workflows/agent-arena-2026-checks.yml` is included. Add repository secrets
`MESH_API_KEY` and `SUBMISSION_TOKEN` (Settings → Secrets and variables → Actions).

---

## 10. Limitations & roadmap

* Single-tenant by default (`current_user_id` dependency is the seam for real auth / multi-user).
* No direct bank aggregation (Account Aggregator / Plaid) — CSV and manual entry only.
* Forecast doesn't model annual seasonality (festivals, bonuses) with < 12 months of history.
* Category budgets, multi-account/credit-card reconciliation and an Alembic migration history are next.
