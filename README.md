# FinPilot — a proactive, multi-agent financial co-pilot

> Most finance apps tell you what already happened. FinPilot works out where your money is **heading**,
> answers *"can I afford this?"* **before** you commit, and tells you **one specific thing to do** — with the
> evidence behind it.

FinPilot ingests a bank statement, and a team of cooperating agents categorises every transaction, studies
behaviour over time, simulates 400 possible futures from your own habits, and turns all of it into ranked,
quantified, *auditable* recommendations. A background scheduler delivers a weekly digest without being asked.

**Every AI call routes through [Mesh API](https://developers.meshapi.ai)** (`https://api.meshapi.ai/v1`) by
default — and the system still runs end-to-end with **zero LLM spend** (each agent has a deterministic path).

**System One + System Two.** Closed-set *decisions* — which category, which chat intent, is this LLM paragraph
faithful, is this goal money you'll spend — go to **[Jev](docs/JEV.md)** (TypeSafe's decision model, via Mesh
`POST /v1/evaluate`): typed answers with calibrated probabilities, billed on input tokens only. LLMs only *write*.
Simple chat questions are answered straight from your data without an LLM; unsafe or off-topic ones are refused
without one.

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
4. [System One decisions with Jev](#4-system-one-decisions-with-jev)
5. [Forecasting & what-if simulation](#5-forecasting--what-if-simulation)
6. [Auditable decision trail](#6-auditable-decision-trail)
7. [Self-evaluation results](#7-self-evaluation-results)
8. [Key design decisions](#8-key-design-decisions)
9. [LLM providers — Mesh by default, bring your own](#9-llm-providers--mesh-by-default-bring-your-own)
10. [API, project layout, deployment](#10-api-project-layout-deployment)
11. [Bank statements — any bank, any format](#11-bank-statements--any-bank-any-format)
12. [Accounts & security](#12-accounts--security)
13. [Observability & debugging](#13-observability--debugging)
14. [Limitations & roadmap](#14-limitations--roadmap)

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

Open the app and **create an account** (e-mail + password) or click **Try the demo** for a private, pre-filled
sandbox. The first account is the administrator, and if you used FinPilot before accounts existed, it keeps that
data. Upload a statement on **Transactions** (CSV, Excel or PDF from any bank), or load a demo persona in
**Settings** (young professional · family with EMIs · freelancer with irregular income). Prefer no login on a
personal machine? `AUTH_ENABLED=false` gives single-user mode, where the demo account is seeded on first boot.

**Frontend dev mode:** `uvicorn finpilot.main:app --reload` + `cd frontend && npm run dev` → http://localhost:5173
(Vite proxies `/api`). **Docker:** `docker compose up --build`. **Make targets:** `make setup | api | web | test | eval`.

**Zero-cost mode.** Without a key (or with `LLM_ENABLED=false`) every agent completes its job deterministically —
93% categorisation accuracy, full forecasting, simulation, advice with template text, and a rule-based chat that
calls the same tools. Adding a Mesh key turns on both systems with the same `rsk_` key: **Jev** decisions
(long-tail categories, chat routing, the faithfulness guard, goal typing — fractions of a cent per thousand) and the
**LLM** for narratives, advice phrasing and complex chat. You can also point the LLM side at a free provider (Groq,
Gemini free tier, OpenRouter `:free`, local Ollama); `JEV_ENABLED=false` turns decisions back to rules.

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
| **Categorizer** | Label every transaction with category + confidence + reason | Jev questions: category (structured criteria), payee type, service family | memory → history → ~200-brand lexicon → keyword evidence (amount- & sign-aware, P2P detection) → recurring-pattern post-pass | **Jev decides each long-tail merchant once** (all in parallel; 300 Swiggy rows never leave the lexicon); unsure answers are deferred, LLM cascade opt-in |
| **Analyzer** | Behaviour over time; patterns & anomalies | Behaviour-profile prompt | rolling 30-day windows vs previous three (z-scores), median/MAD outliers, duplicate/fee/new-payee detection, recurring-series detection, subscription overlap & **usage evidence**, impulse tagging | 3–4 bullet behaviour profile (grounding-checked) |
| **Forecaster** | Project the trajectory; simulate decisions | Forecast-narrative prompt | Monte-Carlo engine (below), goal funding, what-if, "wait until"/"cut" suggestions | Plain-language explanation (grounding-checked) |
| **Advisor** | Turn signals into specific, ranked actions | Advisor prompt (rewrite, never invent numbers) | 11 candidate generators, priority scoring, feedback adaptation, goal-progress adaptation | Friendlier phrasing + weekly headline, rejected if any figure isn't in the evidence |
| **Chat** | Free-form money questions | Jev router (intent, slots, complexity) + system prompt + **10 function-calling tools** (`simulate`, `get_spending`, `get_forecast`, …) | Per-intent handlers that call the *same* tools | Jev answers simple questions from tools and refuses injection/off-topic **without an LLM**; complex ones go to the LLM tool loop |
| **Digest** | Weekly proactive message: one insight + one action | Digest prompt | Template composer | Warm, specific opener (grounding-checked) |

**One run, step by step (demo account):**
1. `plan` — 609 transactions, 609 uncategorised → plan `categorize → analyze → forecast → advise`.
2. `categorize` — 502 by lexicon, 89 by keyword evidence, 18 uncertain → one Jev decision per unknown merchant
   (or the batched LLM when Jev isn't configured).
3. `analyze` — 14 recurring series, 8 subscriptions, 4 anomalies, 12 findings → persisted as insights; anomalies
   and impulse tags written back to transactions.
4. `forecast` — 400 paths × 185 days in ~50 ms; goal snapshots appended (used next run to adapt advice).
5. `advise` — 9 candidates → 8 ranked recommendations, each with trigger + evidence + reasoning.
6. `digest` (scheduled runs) — composed and delivered; per-channel delivery status stored.

---

## 4. System One decisions with Jev

LLMs are for *writing*. Most of what FinPilot asked them to do was *deciding* — pick a category, pick an intent,
judge a paragraph — where the answer is one of a known set. Those decisions now go to **Jev**
(`typesafe/jev-1.13`) through Mesh `POST /v1/evaluate`: you send state + named questions, you get a probability for
every option, already typed. Full design notes, cost analysis and sources: **[docs/JEV.md](docs/JEV.md)**.

| Decision | Before | With Jev |
|---|---|---|
| Unknown merchant → category | batched LLM JSON | `category` + `payee_type` + `service_family` in one request per merchant, all in parallel; stored as a merchant profile and never paid for again |
| Chat message → what to do | LLM tool loop for every message | `intent` (12 options) + slots + `complexity` Score → deterministic tool handler (no LLM), refusal (injection / off-topic / "move my money"), or LLM escalation with a cheaper model for simple ones |
| LLM paragraph → show it? | numeric grounding check | numeric check **+** claim-support Choice (supported / unsupported claim / contradiction / generic) |
| Goal → spent or kept? | all goals treated as savings | `goal_type` Choice + `spend_at_deadline` Noul → trip money leaves the forecast on the deadline |
| Unknown CSV header → column role | rejected | Jev proposes, **code verifies** the values fit (dates parse, amounts are numbers) |
| Unknown subscription → overlap group | lexicon brands only | `service_family` joins lexicon groups (Hoichoi + Netflix = overlapping video streaming) |

**Policy, not blind trust** (`finpilot/decisions/policy.py`): act at ≥ `JEV_ACCEPT_CONFIDENCE`, otherwise keep the
deterministic answer; re-ask mid-confidence answers with the options **reversed** (option-order bias) and average;
log every decision with its probabilities and the action taken (**Decisions** page, `GET /api/decisions`). Math,
dates and amounts never go to Jev — they arrive as code-computed bands ("small: under ₹500", "in about 5 months").
The LLM cascade for unsure categories is **off by default**: an independent benchmark found cascades to a fallback
LLM introduced more errors than they fixed — enable it only if `run_eval --jev` shows a gain.

**Cost** (input tokens only, $0.042/1M): a chat message Jev routes to a tool handler costs ~1/13 of the LLM tool
loop it replaces. Categorization is roughly cost-neutral versus a *batched* cheap LLM — the win there is calibrated
confidence, always-valid labels and latency. The faithfulness guard, goal typing and CSV mapping are new checks and
are reported as added cost, never as savings (`GET /api/decisions/stats`).

| Decisions (Jev) page | Chat routed by Jev |
|---|---|
| ![Decisions](docs/screenshots/decisions.png) | ![Chat routing](docs/screenshots/chat-jev-routing.png) |

<sub>UI previews rendered against a <b>simulated</b> <code>/v1/evaluate</code> (the development sandbox had no network
access to Mesh) — the figures in these two images are illustrative, not measurements. Measured numbers come from
<code>run_eval --jev</code>.</sub>

---

## 5. Forecasting & what-if simulation

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

## 6. Auditable decision trail

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

## 7. Self-evaluation results

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

**System One (Jev):** `python -m evaluation.run_eval --jev` measures every Jev decision on hand-labelled sets
(142 merchants, 62 chat messages incl. Hinglish / injection / off-topic, 24 faithfulness pairs, 24 goals, 20 CSV
columns): accuracy, macro-F1, ECE + reliability table, a threshold chosen on a validation split and reported on a
held-out test split, option-order flip rate, unsafe-message recall and cost per 1k decisions. The live run records a
cassette so `--jev-replay` reproduces it offline. The deterministic baselines are measured in every run — the gaps
Jev has to close:

| Decision | Deterministic path | Measured |
|---|---|---|
| Merchant category | lexicon + keywords | 68.3% overall, **26.9%** on the 52 it's unsure about (those go to Jev) |
| Chat routing | regex router | 50.0%, cannot refuse the 12 unsafe / off-topic / action messages |
| Faithfulness | numeric grounding check | catches **22%** of unfaithful drafts — 14 of 18 contain no wrong number |
| Goal type | keyword table | 75% type · 92% spend-vs-keep |

Jev's own numbers are filled in by a live run (this repository never reports a model number it didn't measure).

---

## 8. Key design decisions

1. **System One decides, System Two writes, code computes.** Closed-set judgments (categories, intents,
   faithfulness, goal types, column roles) go to Jev with a confidence policy and an audit log; LLMs only generate
   text; arithmetic, dates and statistics never leave code. See [docs/JEV.md](docs/JEV.md).
2. **Deterministic-first, LLM where judgment is needed.** Arithmetic, statistics and scheduling are done in code;
   the LLM handles what code can't (unknown merchants, natural language). This keeps the product correct,
   testable and nearly free — and it keeps working when the API key, budget or network doesn't.
3. **Numbers from code, words from the LLM — enforced twice.** LLM-written advice, narratives, digests and chat
   answers pass a numeric grounding check (`finpilot/llm/grounding.py`): any figure not derivable from the evidence
   (with rounding, ×12, /12 tolerance) rejects the text. They then pass a Jev claim-support check
   (`finpilot/decisions/guard.py`) that catches invented causes, wrong merchants and boilerplate the numbers can't
   reveal. Rejections are recorded in the trail and the template is kept.
4. **Pay per merchant, not per row — once.** Uncertain transactions are grouped by merchant; each unknown merchant
   gets one Jev decision (or one line in a 40-merchant LLM batch without Jev), stored as a merchant profile and
   reused as "history" for every future import. Model responses are cached in the DB.
5. **User feedback is a first-class signal.** Category corrections become merchant memory (including prefix
   matches like "Truffles New Branch"); dismissed advice is snoozed 30 days unless the problem grows ≥25%, and
   repeatedly dismissed kinds are down-weighted; goal advice compares with the previous run ("up from 41% → 63%").
6. **Common random numbers for what-if**, so "can I afford it?" isn't answered by noise.
7. **Usage evidence over guesswork.** "Unused subscription" is only claimed when data proves it (a delivery
   membership with no orders from that merchant in 45+ days) or when the user says so — not from a name.
8. **Behaviour ≠ one-offs.** Spikes are computed on variable spend with anomalies and fixed bills removed, so a TV
   purchase or a bill landing twice in a 30-day window doesn't masquerade as a "habit".
9. **One coordinator, explicit graph.** LangGraph gives a readable analyse → forecast → advise flow with
   conditional routing; the step wrapper gives observability and graceful degradation (non-critical node fails →
   run continues; analysis fails → run aborts with the error recorded).
10. **Production basics:** pydantic-settings config, SQLite by default / PostgreSQL via `DATABASE_URL`, idempotent
   statement imports (row fingerprints), per-user accounts and run locks, structured JSON logs with request ids,
   raw model-call traces, Docker image with health check and non-root user, CI (ruff, pytest, evaluation, frontend
   build). Keys live only in env vars — never in the database or the UI.

---

## 9. LLM providers — Mesh by default, bring your own

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
| FreeLLMAPI | `FREELLMAPI_API_KEY` (+ `FREELLMAPI_BASE_URL`) | `auto` | self-hosted router, one unified key; `auto:fast` / `auto:smart` |
| Custom | `CUSTOM_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | `auto` | any OpenAI-compatible server |

* **Per-agent routing:** e.g. `CATEGORIZER_MODEL=openai/gpt-4o-mini`, `ADVISOR_MODEL=anthropic/claude-haiku-4.5`,
  or switch provider/model per agent at runtime on **Settings** (persisted; keys stay in `.env`). The Settings page
  lists Mesh's live model catalogue and has a **Test connection** button.
* **Fallback chain:** `LLM_FALLBACK_PROVIDERS=groq,ollama` — tried in order if the primary fails; if all fail,
  agents fall back to their deterministic path. Every call is logged (tokens, latency, cache hit, success).

**FreeLLMAPI or another unified gateway for local testing:** `LLM_PROVIDER=freellmapi` and
`FREELLMAPI_API_KEY=<your unified key>` (base URL defaults to `http://localhost:3001/v1`, model `auto`), or
`LLM_PROVIDER=custom` with `CUSTOM_LLM_BASE_URL`, `CUSTOM_LLM_API_KEY` and `CUSTOM_LLM_MODEL=auto`. Such gateways
serve chat completions only: Jev decisions need a Mesh `rsk_` key (or `JEV_ENABLED=false` to use rules), and so
does `python -m evaluation.run_eval --jev`, which now checks the key with one tiny call first and explains a
rejected key, missing credits or an unreachable endpoint instead of crashing.

> For the hackathon submission keep `LLM_PROVIDER=mesh` (the default) so that every LLM call goes through Mesh.

---

## 10. API, project layout, deployment

Interactive docs at **`/docs`**. Highlights:

| Endpoint | Purpose |
|---|---|
| `POST /api/auth/signup` · `/login` · `/logout` · `/demo` · `GET /api/auth/session` | accounts (cookie session, or `issue_token` for a bearer token) |
| `POST /api/auth/password/forgot` · `/reset` · `/change` · `/account/delete` | password lifecycle, account deletion |
| `POST /api/transactions/upload` | statement import — CSV / Excel / PDF (optional `password`), idempotent → runs the agents |
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
| `GET /api/decisions` · `GET /api/decisions/stats` | System One decision log (probabilities, action taken) + spend & savings by kind |
| `GET/PUT /api/settings/decisions` · `POST /api/settings/decisions/test` | Jev thresholds, order check, LLM cascade; live connection test |
| `GET /api/evaluation` | self-evaluation results |
| `GET /api/observability/calls[/{id}]` · `/summary` · `/logs` · `/status` | model calls with raw payloads, metrics, log viewer (admin) |
| `POST /api/client-errors` | browser crash reports into the server log |

```
finpilot/
  agents/       coordinator (LangGraph) · categorizer · analyzer · forecaster · advisor · chat · digest
  domain/       taxonomy · merchant lexicon · description normaliser · recurring detection · stats
  decisions/    Jev client (Mesh /v1/evaluate) · question specs · confidence policy · faithfulness guard · cost model
  llm/          provider registry · router (fallback, cache, telemetry) · grounding check
  observability/ request context · access-log middleware · raw call recorder · LangSmith tracing
  services/     statements/ (format readers, column mapping, PDF text parser) · ingestion · auth · synthetic data
                repository · toolkit (chat tools) · scheduler (+ retention) · notifier · seed
  api/          FastAPI routers + serializers        main.py · models.py · config.py · db.py
evaluation/     run_eval.py · jev_eval.py + jev_cases.py (labelled sets) · results/ (REPORT.md, latest.json)
frontend/       React 19 + Vite + TypeScript + Tailwind 4 + Recharts + TanStack Query
tests/          128 tests: domain logic, agents (fake LLM + simulated /v1/evaluate), Jev, statement formats, accounts
                & isolation, observability, API
```

**Deploy for free:** `render.yaml` (Render free web service, Docker), or any Docker host / Hugging Face Docker
Space (`PORT` is honoured). Use a free Postgres (Neon/Supabase) via `DATABASE_URL=postgresql+psycopg://…` for
persistence. Run exactly one instance with `SCHEDULER_ENABLED=true`.

**Hackathon checks:** `.github/workflows/agent-arena-2026-checks.yml` is included. Add repository secrets
`MESH_API_KEY` and `SUBMISSION_TOKEN` (Settings → Secrets and variables → Actions).

---

## 11. Bank statements — any bank, any format

Upload what your bank gives you; FinPilot works out the layout instead of expecting one.

| Format | How it is read |
|---|---|
| CSV / TSV / TXT | delimiter and encoding sniffed; `;` and `\t` exports, cp1252 and UTF-8 with BOM |
| Excel `.xlsx` / `.xls` | openpyxl / xlrd — real date cells, Excel serial dates, and "XLS" files that are really HTML or CSV |
| PDF | pdfplumber tables (ruled and unruled), with a text-line parser for statements without table borders; password-protected PDFs prompt for the password; scanned (image-only) PDFs are reported as such |

* **Header detection** scores every row for date / description / amount / debit / credit / balance vocabulary
  (`Value Date`, `Transaction Remarks`, `Withdrawal Amount (INR)`, `Dr/Cr`, `Narration`…), skipping bank
  letterheads, repeated page headers, opening-balance lines and footers. Files without a header are mapped from the
  column contents; unfamiliar headers can be mapped by Jev, verified against the values.
* **Values:** one date format per column (so 05/07 is never read two ways), `1,23,456.78`, `500 Cr`, `(250)`,
  `250-`, `₹`, `Rs.`, `/-`; remarks wrapped over several lines are joined.
* **Signs** come from separate debit/credit columns, a type column, Cr/Dr suffixes, or the running balance.
* **Checks:** every import reports the date range, money in and out, and how many rows reconcile against the
  statement's own running balance (100% for the ICICI CSV that previously failed). Re-uploading the same file
  imports nothing twice.

## 12. Accounts & security

![Sign in](docs/screenshots/sign-in.png)

* **Sign in with e-mail and password.** Passwords are hashed with scrypt; sessions are random tokens of which only
  a SHA-256 is stored, sent as an `HttpOnly`, `SameSite=Lax` cookie (`Secure` over HTTPS). API clients can request
  a bearer token instead (`issue_token: true`).
* **Every account sees only its own data**: transactions, goals, runs, chats, decisions and model-call traces are
  scoped by user; ids from another account return 404.
* **Admin vs user:** the first account (or anyone in `ADMIN_EMAILS`) administers the server — model providers,
  Jev thresholds, the digest schedule and the raw log viewer, which span all accounts. Everyone manages their own
  profile, data and password.
* **Try the demo:** each visitor gets a private, pre-filled account that is deleted after `DEMO_TTL_HOURS`, so
  judges and testers never see each other's data.
* **Protections:** a CSRF check (`X-Requested-With`) on cookie-authenticated writes and on sign-in; rate limits on
  sign-in, sign-up, password reset and demo creation; constant-time comparisons and no account enumeration;
  password change and reset sign out other sessions; security headers on every response.
* **Password reset** e-mails a one-hour, single-use link through the SMTP settings. Without SMTP the link is
  written to the server log for the operator to pass on.
* `AUTH_ENABLED=false` restores single-user mode (no login; optional static `API_TOKEN`).

## 13. Observability & debugging

* **Every model call is recorded** (LLM and Jev, live or cached, success or failure) with its **raw request and
  response**: the exact messages and tools sent, the provider's full JSON reply, tokens, latency, cost, the agent
  run, the user and the HTTP request that caused it. Browse them on the **Observability** page: filter by kind,
  status or text, and click a call to see the prompt, the answer and the raw JSON. Payloads stay in your database
  (`TRACE_PAYLOADS`, clipped to `TRACE_MAX_CHARS`, purged after `TRACE_RETENTION_DAYS`).
* **Structured logs:** a readable console plus a rotating JSON-lines file (`LOG_FILE`, default
  `logs/finpilot.log`). Every line carries `request_id`, `user_id` and the agent `run_id`, and every request gets an
  access-log line with its status and duration. Admins can search and live-tail the file on the Observability page;
  `LOG_LEVEL=DEBUG` also logs raw model payloads.
* **Request ids:** every response has an `X-Request-ID` header (an upstream proxy's id is kept). Server errors
  return the id, so one id finds the failing request, its log lines and the model calls it made. Browser crashes are
  reported to `/api/client-errors` and land in the same log.
* **LangSmith (optional):** `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=…` trace the LangGraph pipeline (each
  agent a span), every OpenAI-SDK call, Jev decisions and chat turns as nested runs in project `LANGSMITH_PROJECT`.
  It is off by default because traces contain financial data and leave your machine.

| Model calls, latency and failures | One call: the prompt, the answer, the raw JSON |
|---|---|
| ![Observability](docs/screenshots/observability.png) | ![Call trace](docs/screenshots/call-trace.png) |

```bash
tail -f logs/finpilot.log | jq 'select(.level != "DEBUG") | {ts, level, request_id, user_id, msg}'
jq 'select(.request_id == "3f9c0e…")' logs/finpilot.log   # everything one request did
```

## 14. Limitations & roadmap

* Accounts are e-mail + password: no SSO, two-factor sign-in or e-mail verification yet, and the rate limiter is
  per process (put a shared limiter or the reverse proxy in front of several replicas).
* No direct bank aggregation (Account Aggregator / Plaid) — statement upload (CSV, Excel, PDF) and manual entry only. Scanned (image-only) PDFs need OCR first.
* Forecast doesn't model annual seasonality (festivals, bonuses) with < 12 months of history.
* Category budgets, multi-account/credit-card reconciliation and an Alembic migration history are next.
* Mesh's `/v1/evaluate` is a **beta** endpoint whose contract may change; the client pins `X-Mesh-Version` and
  validates requests locally, and every Jev decision has a deterministic fallback. Jev accuracy/calibration figures
  are pending a live `run_eval --jev` (the development sandbox could not reach the API).
