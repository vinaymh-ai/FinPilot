# FinPilot — a proactive, multi-agent financial co-pilot

> Most finance apps tell you what already happened. FinPilot works out where your money is **heading**,
> answers *"can I afford this?"* **before** you commit, and tells you **one specific thing to do** — with the
> evidence behind it.

FinPilot ingests a bank statement, and a team of cooperating agents categorises every transaction, studies
behaviour over time, simulates 400 possible futures from your own habits, and turns all of it into ranked,
quantified, *auditable* recommendations. A background scheduler delivers a weekly digest without being asked.

**Every AI call routes through [Mesh API](https://developers.meshapi.ai)** (`https://api.meshapi.ai/v1`) by
default — and the system still runs end-to-end with **zero LLM spend** (each agent has a deterministic path).

**Ask anything about your transactions.** *Ask FinPilot* is its own team of agents — guardrails, a retriever
(value linking + knowledge RAG), a **sandboxed per-user SQL Analyst**, a planner over the finance tools, and a
composer + verifier — so "how many transactions did I make?", "show all my shopping transactions", "UPI vs
card spending", "biggest credit this year" or "food delivery this month vs last month" get exact answers with
the full table attached. Measured: **96.4% end-to-end answer accuracy** against ground truth, and every
transaction carries its **debit/credit direction and payment channel** (UPI, card, NEFT, ATM, auto-debit…).

**It remembers — and you are in control of categories.** Chats are threads like ChatGPT (new chat, history,
search, pin, rename). A **Memory agent** resolves follow-ups ("the same person", "and in August?", "both"),
recaps the conversation and keeps what you tell it ("Hanumanthaiah is my landlord") across chats. Retrieval is
**semantic** ("coffee" → Starbucks, "cab rides" → Uber/Ola/Rapido, "hanumantha" → both spellings of that payee).
You can create **your own categories**, write **rules** with a live preview, bulk-review uncertain
transactions, and see **every category change** a transaction went through.

**API agents with verified evidence.** Separate categorizer, analyzer, forecaster, advisor, auditor, chat
planner, SQL analyst and digest prompts reason over real transaction evidence. Exact tools compute money; independent API
checks reject unsupported generated claims. Missing balances limit affordability conclusions while
cash-flow projections and transaction-based advice continue. **Jev is disabled by default**; its optional
legacy integration remains documented below. See the current [agent design](docs/API_AGENTS.md).

![Overview](docs/screenshots/overview.png)

| What-if Lab | Decision trail ("Why?") |
|---|---|
| ![What-if](docs/screenshots/what-if-lab.png) | ![Decision trail](docs/screenshots/decision-trail.png) |
| **Ask FinPilot: conversation threads + memory (“the same person”, “and in the last 30 days?”)** | **Recap of the chat; long-term memory turned into a confirmed recategorisation** |
| ![Chat](docs/screenshots/chat.png) | ![Chat recap](docs/screenshots/chat-recap.png) |
| **Categories & rules: your own categories, rules, review queue** | **Rule preview before saving** |
| ![Categories](docs/screenshots/categories.png) | ![Rule preview](docs/screenshots/rule-preview.png) |
| **Transactions: bulk review (approve / move), debit/credit, channel** | **Category history: every change, by whom and why** |
| ![Transactions](docs/screenshots/transactions.png) | ![Category history](docs/screenshots/category-history.png) |
| **Agents: analysis pipeline + chat team** | **Self-evaluation** |
| ![Agents](docs/screenshots/agents.png) | ![Evaluation](docs/screenshots/evaluation.png) |

---

## Contents
1. [What it does — the story](#1-what-it-does--the-story)
2. [Quick start](#2-quick-start)
3. [How the agents cooperate](#3-how-the-agents-cooperate) · [3a. Ask FinPilot — the chat team](#3a-ask-finpilot--the-chat-team) · [3b. Categories you control](#3b-categories-you-control)
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
| **Conversational** | Any question about your own transactions — "How many transactions did I make last month?", "Show all my shopping transactions", "Average Swiggy order", "Spending by category this year", "UPI vs card", "Food delivery this month vs last month" — answered by a sandboxed SQL Analyst with the exact table/chart attached; plus "Am I on track?", "Can I afford a ₹1.2L bike in December?" through the finance tools. Every answer shows which agents and queries produced it. |
| **Trust** | Every transaction stores direction (debit/credit), payment channel, category source, confidence and reason. An **Auditor** agent re-checks every recommendation's trigger, evidence transactions and figures before you see it. |

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
personal machine? `AUTH_ENABLED=false` gives single-user mode. With `SEED_DEMO_DATA=true`, a new database is
populated on first boot, and the first signup inherits the starter data. Subsequent accounts start empty.

**Frontend dev mode:** `uvicorn finpilot.main:app --reload` + `cd frontend && npm run dev` → http://localhost:5173
(Vite proxies `/api`). **Docker:** `docker compose up --build`. **Make targets:** `make setup | api | web | test | eval`.

### Docker with PostgreSQL

Requires Docker Compose 2.24+ (the `.env` file is optional). Run:

```bash
docker compose up --build -d
```

Open http://localhost:8000. Compose creates the `finpilot` PostgreSQL database, waits for it to become healthy,
then the app creates its tables and loads eight months of sample transactions, goals, forecasts and advice.
Initial analysis uses deterministic agents, so startup requires no AI credentials. Create your first account
to access that starter data, or use **Try the demo** for a private sandbox. Restarting preserves existing data
and edits; set `SEED_DEMO_DATA=false` before the first start if you want an empty database.

Database data persists in the `pg-data` volume. `docker compose down` keeps it. Existing SQLite files in
`finpilot-data` remain available, but are not automatically imported into PostgreSQL.
Compose uses its PostgreSQL service regardless of `DATABASE_URL`, which applies to runs outside Compose.
Default local connection details (also usable in pgAdmin/DBeaver):

| Host | Port | Database | Username | Password |
| --- | --- | --- | --- | --- |
| `localhost` | `55432` | `finpilot` | `finpilot` | `finpilot` |

Configure `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` and `POSTGRES_PORT` in `.env` before the first start
to override those defaults. Use a URL-safe password. For a Python backend running on your host, set
`DATABASE_URL=postgresql+psycopg://finpilot:finpilot@localhost:55432/finpilot`.

### FreeLLMAPI connection

FreeLLMAPI must be running separately with working provider keys and its unified API key. In `.env`, set:

```dotenv
LLM_PROVIDER=freellmapi
LLM_MODEL=auto
FREELLMAPI_API_KEY=your-unified-key
FREELLMAPI_BASE_URL=http://localhost:3001/v1
```

Compose overrides the gateway URL to `http://host.docker.internal:3001/v1`, because `localhost` inside the
app container refers to that container. For a remote gateway or one on a shared Docker network, set
`DOCKER_FREELLMAPI_BASE_URL` to its reachable URL instead. After changing `.env`, run
`docker compose up -d` to recreate the app with the new environment, then use **Settings → Test connection**.

### Semantic search for chat (optional embeddings)

Chat links your words to payees and categories with a **local hybrid index** that needs no API: Indian-name
spelling variants ("hanumantha" → Hanumanthaiah *and* Hanumanthiah), TF-IDF vectors over words and character
n-grams, and *meaning* features (category vocabulary plus what each brand or local business is — "coffee" →
Starbucks, "cab rides" → Uber/Ola/Rapido). An embedding model adds dense vectors on top, fused by Reciprocal
Rank Fusion. Any OpenAI-compatible `/v1/embeddings` works — FreeLLMAPI included:

```dotenv
# through your FreeLLMAPI key (pin a family from its Models → Embeddings page; bge-m3 also works)
EMBEDDING_PROVIDER=freellmapi
EMBEDDING_MODEL=gemini-embedding-001
# …or free and local:  EMBEDDING_PROVIDER=ollama  EMBEDDING_MODEL=nomic-embed-text  (ollama pull nomic-embed-text)
# …or OpenAI:          EMBEDDING_PROVIDER=openai  EMBEDDING_MODEL=text-embedding-3-small  OPENAI_API_KEY=…
```

Pin one model rather than `auto`: vectors from different models cannot be compared, so if a gateway's default
changed under `auto`, dense retrieval would switch itself off. Payees are embedded in batches of 64 and cached per
model, so after the first question only the new question is embedded (well within a 1,000-requests-a-day free
tier). Leave `EMBEDDING_PROVIDER` blank to reuse `LLM_PROVIDER`. Compose reaches Ollama on your machine through
`http://host.docker.internal:11434/v1` (`DOCKER_OLLAMA_BASE_URL`). Check the setup with **Settings → Test
embeddings**. If the endpoint fails or takes longer than `EMBEDDING_TIMEOUT_S` (8 s), chat skips dense retrieval
for five minutes and answers from the local index, so a dead endpoint never slows the chat down.

Jev decisions use a separate API. For a direct TypeSafe connection, set `JEV_PROVIDER=typesafe`,
`TYPESAFE_API_KEY` and `JEV_MODEL=jev-1.13.0`. The shorter `jev-1.13` id is rejected by TypeSafe; the client now
translates Mesh's default alias for direct calls. This optional legacy route is off by default;
`JEV_ENABLED=false` keeps decisions on the configured LLM API.

**Zero-cost mode.** Without a key (or with `LLM_ENABLED=false`) every agent completes its job deterministically —
rule-based categorization, forecasting and simulation with a verified balance, template advice, and chat that
calls the same tools. Adding a Mesh key enables API reasoning for categories, query planning, advisor
prioritization, goal typing, narratives and independent claim checks. FreeLLMAPI serves development
calls; submission calls must use Mesh. Jev stays disabled unless explicitly selected as an optional route.

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
    advise --> audit
    audit -- scheduled run --> digest --> E
    audit --> E
```

| Agent | Its job | Own prompt / tools | Deterministic path | Where the LLM adds value |
|---|---|---|---|---|
| **Coordinator** | Plans the run (skip categorisation if nothing is new, add digest on scheduled runs), routes, records the trail | LangGraph conditional edges | always | — |
| **Categorizer** | Label every transaction with category + confidence + reason | Full narration, direction, amount, dates and related transactions; built-in **and your own** categories; your corrections as few-shot conventions | **your rules** → your corrections → trusted history → lexicon → keyword evidence → P2P detection | LLM reasoning for unresolved purposes, including payment platforms; Jev handles remaining uncertainty; provider payments with different descriptions stay separate. Every change is logged per transaction ([§3b](#3b-categories-you-control)) |
| **Analyzer** | Behaviour over time; patterns & anomalies | Behaviour-profile prompt | rolling 30-day windows vs previous three (z-scores), median/MAD outliers, duplicate/fee/new-payee detection, recurring-series detection, subscription overlap & **usage evidence**, impulse tagging | 3–4 bullet behaviour profile (grounding-checked) |
| **Forecaster** | Project the trajectory; simulate decisions | Forecast-narrative prompt | Monte-Carlo engine (below), goal funding, what-if, "wait until"/"cut" suggestions | Plain-language explanation (grounding-checked) |
| **Advisor** | Turn signals into specific, ranked actions | Advisor prompt (rewrite, never invent numbers) | 11 candidate generators, priority scoring, feedback adaptation, goal-progress adaptation | Friendlier phrasing + weekly headline, rejected if any figure isn't in the evidence |
| **Auditor** | Verify the decision trail | Own checks over this run's findings, forecast and ledger | always (no LLM needed) | — (deliberately independent of the models it audits) |
| **Chat team** | Free-form money questions, follow-ups, recaps | Memory · Guardrails · Retriever · Planner (`query_ledger`, `find_payees`, `get_spending`, `simulate`, …) · SQL Analyst · Composer/Verifier — see [§3a](#3a-ask-finpilot--the-chat-team) | Deterministic question parser → SQL, finance tools, templates | Planning, text-to-SQL with self-repair, natural answers; numeric and claim checks reject unsupported details |
| **Digest** | Weekly proactive message: one insight + one action | Digest prompt | Template composer | Warm, specific opener (grounding-checked) |

**One run, step by step (demo account):**
1. `plan` — 609 transactions, 609 uncategorised → plan `categorize → analyze → forecast → advise`.
2. `categorize` — confirmed rules handle known merchants; unresolved narrations reach batched LLM reasoning,
   then Jev for remaining uncertainty. Confidence and the classification reason are retained.
3. `analyze` — 14 recurring series, 8 subscriptions, 4 anomalies, 12 findings → persisted as insights; anomalies
   and impulse tags written back to transactions.
4. `forecast` — 400 paths × 185 days in ~50 ms; goal snapshots appended (used next run to adapt advice).
5. `advise` — 9 candidates → 8 ranked recommendations, each with trigger + evidence + reasoning.
6. `digest` (scheduled runs) — composed and delivered; per-channel delivery status stored.

---

## 3a. Ask FinPilot — the chat team

```mermaid
graph LR
    Q([message + conversation]) --> M[Memory: resolve follow-ups, clarifications, recaps, facts]
    M -- recap / remember / confirm --> MA([answered by memory])
    M --> G{guardrails}
    G -- action / injection / off-topic --> R([refusal])
    G -- greeting / help --> C([capabilities from your data])
    G --> RET[Retriever: payees by name, spelling and meaning; categories; insights]
    RET --> P{understood exactly?}
    P -- yes --> SQL[SQL Analyst: compiled query]
    P -- no --> PL[Planner LLM]
    PL --> T[finance tools: spending, forecast, simulate, goals, advice]
    PL --> SQL2[SQL Analyst: LLM text-to-SQL + repair]
    SQL --> A([answer + table/chart])
    SQL2 --> CV[Composer + Verifier] --> A
    T --> CV
```

| Agent | Job | How it stays accurate |
|---|---|---|
| **Memory** | Working memory of each conversation (what the last answer was about, open questions) and long-term memory of what you taught it ("Hanumanthaiah is my landlord", "these two spellings are one person", "call me Vinay") | Resolves references ("the same person", "her", "those payments"), ellipsis ("and in August?", "what about Ola?" — the previous structured query with one aspect swapped), replies to clarifications ("both", "the first one"), list follow-ups ("list please", "which ones were credits?"), recategorisation in chat ("mark X as family", "change the category to Rent" — after a **yes**, creating the category if it is new), recaps; deterministic, so follow-ups are reproducible. Facts are only saved from explicit statements, never from prompt-injection text, and can be listed/deleted in chat or **Settings → Memory** |
| **Guardrails** | Refuse money-moving actions, prompt injection, off-topic chat | Runs before any model sees the message |
| **Retriever (RAG)** | Link words to *your* payees and categories — by spelling ("swigy" → Swiggy; "hanumantha" → Hanumanthaiah **and** Hanumanthiah, recognised as one payee), by meaning ("coffee" → Starbucks, "cab rides" → Uber/Ola/Rapido, "pizza" → Domino's, "petrol" → Shell + Sri Durga Fuels) and by memory ("my landlord"); pull relevant insights, recommendations and goals | **Hybrid semantic search**: TF-IDF sparse vectors over words, character n-grams and *meaning* (category vocabulary + merchant enrichment: what each brand or local business *is*), BM25 over narrations, an Indian-name consonant skeleton for spelling variants, plus **optional dense embeddings** (`EMBEDDING_MODEL`, any OpenAI-compatible `/embeddings`) fused by Reciprocal Rank Fusion. Works offline and reproducibly without the embedding API |
| **SQL Analyst** | Any factual question: counts, lists, averages, max/min/first/last, breakdowns by month/category/merchant/weekday/channel, credits & income, period comparisons, weekend/late-night, amount thresholds | Runs in a **per-request in-memory SQLite that contains only your rows** (authorizer allows SELECT only, no PRAGMA/ATTACH, CPU/size limits). A deterministic parser compiles fully understood questions; anything else goes to LLM text-to-SQL with schema, date anchors and retrieved values, repaired from the database error up to twice. Unknown payees give "no matches", never a grand total |
| **Planner** | Pick the specialist tools for everything else (`get_spending`, `rank_spending`, `simulate`, `get_forecast`, `get_goals`, `get_recommendations`, `get_insights`, `query_ledger`) | Only payees that appear in the question or in your data are accepted |
| **Composer + Verifier** | Phrase the answer from verified results | Numeric grounding + an independent claim check; on rejection the exact deterministic answer is shown |

Results come back as text **plus artifacts**: the full table of matching rows (CSV export) and a chart for
time series — so "give me the transactions related to shopping" shows all of them, not the first ten.

**Conversations like ChatGPT.** Every chat is a thread with an automatic title; the sidebar lists them by date
(pinned first), with search across titles and messages, rename, pin and delete, and **New chat**. A conversation
keeps its working memory across reloads (it is stored with each answer), so this works:

> **You:** how much have I sent to hanumantha? → **₹25,500** to Hanumanthaiah / Hanumanthiah (*2 spellings of the same payee*)
> **You:** how much have I received from the same person? → **₹1,500** (*memory: “the same person” → Hanumanthaiah and Hanumanthiah*)
> **You:** and in the last 30 days? → the same question, new period
> **You:** Hanumanthaiah is my landlord → saved; *“Shall I move 8 transactions (₹27,000, now under Transfers) to Rent & Housing?”* → **yes** → done, logged, and future imports follow
> **You** (new chat, next week): how much did I pay my landlord this year? → answered from long-term memory
> **You:** summarise our chat → a recap of every question with its key figure
> **You:** mark Hanumanthaiah as family → *“You don't have a Family category yet. Shall I create it (it counts like Transfers) and move 10 transactions into it?”* → **yes** → created, moved, remembered
> **You:** how many transactions relate to rahul? → **list please** → **which ones were credits?** → each answer reuses the previous question, so short follow-ups never need re-explaining

---

## 3b. Categories you control

The categorizer is a cascade with an explanation on every row; around it, FinPilot gives the user the controls the
best transaction-classification products have (we studied 75way's, Trelium's, ZBrain's and HelloBooks'
categorization agents):

| Capability | What FinPilot does |
|---|---|
| **Your own categories** | Create "Household help", "Pet care", "Kids' school"… Each one *behaves like* a built-in parent you pick, so analytics stay correct (spending vs transfers, essential vs lifestyle, refunds). The categorizer's model sees your categories and their descriptions; chat and the SQL view know them by name |
| **Your rules** | "Payee contains KAMALA → Household help", "narration has the word SCHOOL and amount ≥ 10,000 → Education", debit/credit and channel filters. Rules run **first**, apply to existing rows on save (with a live preview of what changes) and to every future import. No user regex — terms are matched as words |
| **Learning from corrections** | Fix one transaction or a merchant once; it becomes merchant memory *and* a few-shot convention the model follows for similar payees |
| **Review queue** | Anything below 70% confidence is flagged instead of guessed; review it with **bulk approve / bulk move** (optionally "remember these merchants") |
| **Audit trail** | Every category change — categorizer run, your rule, your edit, a bulk review, a chat confirmation — is recorded with before/after, source, confidence and reason (`GET /api/transactions/{id}/history`, the ⟲ button on each row) |
| **Chat-driven fixes** | "Mark Hanumanthaiah as rent" or "Priya is my sister" → FinPilot proposes the change, you confirm with "yes" |
| **Any kind of bank transaction** | Salary, dividends, interest, tax payments/refunds (new **Taxes** category), bank charges and GST on charges, reversals and refunds, self-transfers, wallet loads, SIPs/FDs/RDs/NPS/PPF/digital gold, EMIs, insurance, card bills, rent platforms, FASTag/metro/fuel, school fees, donations — measured on a second hand-labelled benchmark across 11 banks ([§7](#7-self-evaluation-results)) |

How we used the references: a findings ledger (REA's method — *observations vs inferences vs unknowns*, each
claim tied to evidence) over the four product pages, then a gap checklist. Their pages describe *what* the products
do (rules + history + AI, confidence-gated review, correction memory, user taxonomies, audit trails), not their
internals; split transactions and receipt matching are left on the roadmap.

---

## 4. System One decisions with Jev

**Optional legacy integration:** the active default uses LLM API reasoning and API fact checking,
with `JEV_ENABLED=false`. This section describes the retained opt-in implementation.

The LLM supplies semantic reasoning and natural-language answers. **Jev** supplies closed-set routing,
remaining category decisions and checks of whether generated claims are supported
(`typesafe/jev-1.13`) through Mesh `POST /v1/evaluate`: you send state + named questions, you get a probability for
every option, already typed. Full design notes, cost analysis and sources: **[docs/JEV.md](docs/JEV.md)**.

| Decision | Before | With Jev |
|---|---|---|
| Unknown merchant → category | rule fallback | LLM reasons over unresolved narrations; Jev supplies category/payee/service-family decisions where uncertainty remains |
| Chat message → what to do | fixed templates | Routing selects exact retrieval or a tool loop, followed by an LLM answer with claim checks; greetings need no ledger retrieval |
| LLM paragraph → show it? | numeric grounding check | numeric check **+** claim-support Choice (supported / unsupported claim / contradiction / generic) |
| Goal → spent or kept? | all goals treated as savings | `goal_type` Choice + `spend_at_deadline` Noul → trip money leaves the forecast on the deadline |
| Unknown CSV header → column role | rejected | Jev proposes, **code verifies** the values fit (dates parse, amounts are numbers) |
| Unknown subscription → overlap group | lexicon brands only | `service_family` joins lexicon groups (Hoichoi + Netflix = overlapping video streaming) |

**Policy, not blind trust** (`finpilot/decisions/policy.py`): act at ≥ `JEV_ACCEPT_CONFIDENCE`, otherwise keep the
deterministic answer; re-ask mid-confidence answers with the options **reversed** (option-order bias) and average;
log every decision with its probabilities and the action taken (**Decisions** page, `GET /api/decisions`). Math,
dates and amounts never go to Jev — they arrive as code-computed bands ("small: under ₹500", "in about 5 months").
Initial LLM reasoning runs whenever the categorizer's LLM is enabled. The optional cascade setting controls an
additional retry after Jev defers; it does not disable the initial reasoning pass.

**Cost:** model usage is recorded per run and step. Exact retrieval avoids repeated tool-selection calls for
straightforward questions, while answer generation and validation still use models. Faithfulness checks,
goal typing and CSV mapping are reported as added cost (`GET /api/decisions/stats`).

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

**Categorisation (ablation, deterministic):**

| Pipeline stage | Accuracy | Macro-F1 | Long-tail merchants |
|---|---|---|---|
| Baseline: always the most common category | 20.0% | – | – |
| Merchant lexicon only | 79.0% | 0.827 | 3.8% |
| + keyword phrases, P2P detection & amount-aware rules | **96.3%** | **0.940** | **83.2%** |
| + LLM for uncertain merchants (`--llm`) | run with `MESH_API_KEY` | | |

**Precision when confident:** 91.3% of transactions are labelled with confidence ≥ 0.7 and **99.73%** of those
are correct; the rest go to the *Needs review* queue, and the advisor asks you to confirm them (each answer
becomes merchant memory). Payment channel 99.8% · debit/credit 100%. The 34 long-tail merchants are a frozen
hold-out: the evaluation fails loudly if any of them ever leaks into the lexicon.

**Real bank narrations** (`evaluation/narration_cases.py`, 100 hand-labelled HDFC/ICICI/SBI/Axis/Kotak
formats, independent of the generator): first held-out run **91.0%** category / 97.0% channel / 98.8% precision
when confident; after fixing the generic parsing bugs it exposed, 98.0% / 99.0% / 100%.

**Every kind of bank transaction** (`evaluation/narration_cases_2.py`, 120 hand-labelled narrations from 11 banks —
salary/dividend/interest, tax payments and refunds, bank charges and GST on charges, reversals, self-transfers,
wallet loads, SIP/FD/RD/NPS/digital gold, EMIs, insurance, card bills, rent platforms, FASTag/metro/fuel, school
fees, donations): first run **85.0%** category / 96.5% channel / 96.9% precision when confident. It exposed
missing *kinds* of transaction (no Taxes category, bank-charge wording, self-transfers, wallet loads, deposits,
the EAW ATM code) — after adding them, **100%** / 100% / 100% with 95.8% of rows confident. Both numbers are
reported because the second is measured on the set that motivated the fixes.

**Ask FinPilot end-to-end:** **96.4%** of 28 questions answered with exactly the right figures, where the
expected values are computed from the generator's ground-truth categories, merchants and times (the one miss
is a long-tail shop the deterministic categorizer leaves for review). Tool routing: 20/20.

**Conversation memory:** 4 multi-turn chats (10 turns) where later turns only make sense with memory — “and in
August?”, “what about dining?”, “how many times did I pay her?”, “what about Ola?”, “summarise our chat”: **100%**
of turns correct (follow-ups 6/6). First run, before fixing what it exposed (ellipsis after a spending-tool answer
had no structured focus to inherit): 80% overall, 67% of follow-ups.

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
size on 6/6 datasets.

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
| `POST /api/chat` | multi-agent assistant: answer + tool/SQL trace + table/chart `artifacts` + agent `pipeline` + `memory` (what earlier turns resolved, facts saved); omit `session_id` to start a new conversation |
| `GET /api/chat/sessions?q=` · `PATCH`/`DELETE /api/chat/sessions/{id}` · `GET /api/chat/{id}` | conversation threads: list/search, rename, pin, delete, history |
| `GET /api/chat/memory` · `DELETE /api/chat/memory[/{id}]` | long-term memory the chat keeps about you |
| `GET/POST/PATCH/DELETE /api/categories[/{id}]` | your own categories (each behaves like a built-in parent) + counts, rules and the review count |
| `POST /api/category-rules/preview` · `GET/POST/PATCH/DELETE /api/category-rules[/{id}]` | categorization rules with a live preview; applied to existing rows and future imports |
| `POST /api/transactions/bulk` · `GET /api/transactions/{id}/history` | bulk approve / move from the review queue; per-transaction category audit trail |
| `GET /api/transactions?direction=debit\|credit&channel=upi` | filter by money in/out and payment rail; response includes debit/credit totals |
| `GET /api/digests` · `POST /api/digests/send-now` · `PUT /api/scheduler` | proactive delivery |
| `GET/PUT /api/settings…` · `GET /api/llm/models` · `POST /api/settings/llm/test` | profile & provider routing |
| `GET /api/decisions` · `GET /api/decisions/stats` | System One decision log (probabilities, action taken) + spend & savings by kind |
| `GET/PUT /api/settings/decisions` · `POST /api/settings/decisions/test` | Jev thresholds, order check, LLM cascade; live connection test |
| `GET /api/evaluation` | self-evaluation results |
| `GET /api/observability/calls[/{id}]` · `/summary` · `/logs` · `/status` | model calls with raw payloads, metrics, log viewer (admin) |
| `POST /api/client-errors` | browser crash reports into the server log |

```
finpilot/
  agents/       coordinator (LangGraph) · categorizer · analyzer · forecaster · advisor · auditor · digest
                chat (guardrails, planner, composer/verifier) · analyst (SQL Analyst: parser + text-to-SQL)
                memory (conversation focus, follow-ups, recaps, long-term facts)
  domain/       taxonomy (+ custom categories) · merchant lexicon · enrichment · user rules · normaliser · recurring
  decisions/    Jev client (Mesh /v1/evaluate) · question specs · confidence policy · faithfulness guard · cost model
  llm/          provider registry · router (fallback, cache, telemetry) · grounding check
  observability/ request context · access-log middleware · raw call recorder · LangSmith tracing
  services/     statements/ (format readers, column mapping, PDF text parser) · ingestion · auth · synthetic data
                ledger_sql (sandboxed per-user SQL view) · retrieval (value linking + BM25 knowledge)
                semantic (hybrid sparse vectors, name variants, optional dense embeddings + RRF)
                repository · toolkit (chat tools) · scheduler (+ retention) · notifier · seed
  api/          FastAPI routers + serializers        main.py · models.py · config.py · db.py
evaluation/     run_eval.py · jev_eval.py + jev_cases.py (labelled sets) · results/ (REPORT.md, latest.json)
frontend/       React 19 + Vite + TypeScript + Tailwind 4 + Recharts + TanStack Query
tests/          290 tests: domain logic, agents (fake LLM + simulated /v1/evaluate), Jev, statement formats, accounts
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
  Every stored transaction then records its **direction** (debit/credit) and **payment channel** (UPI, card,
  net banking, NEFT, IMPS, RTGS, ATM, auto-debit, cheque, cash deposit, bank interest/charges) read from the
  narration; both are filters on the Transactions page and columns in the chat's SQL view.
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

## 15. Accuracy and evidence

Transactions expose two levels: `transaction_kind` (expense, refund, transfer, credit-card repayment,
investment, withdrawal, income or unknown) and the spending category. `payment_provider` separately
identifies intermediaries such as CRED, Kiwi or Razorpay. An app name or a person's name alone does not
establish what a payment was for. Explicit card bills use **Credit Card Repayments**; ambiguous descriptors
stay **Other / Needs review**. Use the Transactions review filter to confirm them. Corrections to a payment
provider apply to that individual transaction, since the next payment through the app can have a different
purpose. A normal analysis run also repairs legacy automatic gateway labels; manual and imported labels stay.

Merchant totals and transaction searches take the name and date filters from the question and execute over
the signed-in user's records. Named-merchant questions without dates use all available statement history;
answers show the covered dates, counts and matching IDs. Whole-name matching avoids treating `Ram` as
`Ramesh`. Missing matches produce a clear no-match answer rather than another category's total. Tool traces
retain the actual results, including search truncation and refunds.

LLM financial text requires successful retrieval, numeric grounding and a supported faithfulness verdict.
Unverified drafts are discarded and deterministic handlers answer instead; a warning never makes an
unsupported answer acceptable. This applies to chat and agent narratives. With Jev unavailable, financial
narratives use deterministic templates.

Once real statements are present, demo transactions, linked subscriptions and marked starter goals are
excluded from the real-data view and agents, while remaining stored. A real statement replaces the demo
balance anchor; without a supplied balance, affordability checks request a verified balance in Settings.
Editing a marked starter goal explicitly adopts it as your own goal. Old databases may contain goals or
balances with no provenance, which cannot safely be identified as sample data automatically.

The planner stops on failed classification, analysis or forecasting, and invalidates old outputs when the
inputs change. Analysis reports uncovered categories and available history; short histories are marked as
limited evidence. Forecasts remain estimates of future behavior, never guaranteed balances. Repayments are
reported as financial outflows; this does not automatically reconcile duplicate bank and card exports.

Backend tests default to an isolated SQLite database. To exercise PostgreSQL, point
`FINPILOT_TEST_DATABASE_URL` at a **dedicated empty test database**; the suite drops and recreates its tables.
