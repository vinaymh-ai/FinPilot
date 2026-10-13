# System One decisions with Jev

The current flow combines LLM reasoning, typed decisions and code calculations. Unresolved bank narrations
reach the LLM before Jev handles remaining uncertainty. Financial chat replies use LLM generation from exact
retrievals; greetings need no tools. Claim checks and numeric validation protect the answers. The optional
categorization cascade controls a retry after Jev, not the initial LLM pass.

See [agent accuracy design](AGENT_ACCURACY.md). The original design and its cost comparisons below describe
the earlier Jev-first approach; calls following LLM reasoning are now recorded separately and do not count
as avoided LLM calls.

## Original design notes

FinPilot used LLMs for two different kinds of work:

* **Writing** — phrasing advice, explaining a forecast, answering an open question. Only a generative model can do this.
* **Deciding** — which category is "POS 4495… TRUFFLES BENGALURU"? Is this chat message a spending question, a
  prompt injection, or something that needs real reasoning? Does this LLM paragraph claim anything the evidence
  doesn't support? The answer is always **one of a fixed set of options**.

Closed-set decisions benefit from structured validation. LLM output must be parsed and checked against the option
set, and its self-reported confidence is not calibrated. **Jev** (TypeSafe's "System One" model,
`typesafe/jev-1.13`) is built for exactly this kind of decision. You send `state` plus named questions. Back comes
a probability for every option, already typed, and you are billed only for input tokens.

FinPilot now splits the work: **Jev decides, LLMs write, code computes.** Every Jev call goes through Mesh
`POST /v1/evaluate`, so the hackathon rule ("every AI call routes through Mesh") still holds.

```
                 ┌──────────── System One (Jev, /v1/evaluate) ────────────┐
 statement row → │ category · payee type · service family  (per merchant) │ → rules / profiles
 chat message  → │ intent · slots · complexity · injection/off-topic      │ → tool handler | refusal | LLM
 LLM draft     → │ supported / unsupported / contradiction / generic       │ → show | template
 goal name     → │ goal type · P(spent at deadline)                        │ → forecast outflow
 CSV header    → │ date / description / amount / debit / credit / …        │ → verified mapping
                 └──────────────────────────────────────────────────────────┘
                 ┌──────────── System Two (LLM, /v1/chat/completions) ────┐
                 │ advice phrasing · narratives · digest · complex chat    │
                 └──────────────────────────────────────────────────────────┘
                 ┌──────────── Code ──────────────────────────────────────┐
                 │ amounts, dates, statistics, Monte-Carlo, thresholds     │
                 └──────────────────────────────────────────────────────────┘
```

## Where Jev fits — the decision matrix

| # | Decision | Before | Now | Question(s) | If Jev is unsure / off |
|---|---|---|---|---|---|
| 1 | **Category of an unknown merchant** (`categorizer`) | Batched LLM JSON, 40 merchants per call, self-reported confidence | One request per unique merchant, all merchants in parallel; reused forever via `merchant_profiles` | `category` Choice (22 options, structured `what`/`not_for`), `payee_type` Choice, `service_family` Choice (debits under ₹5k) | rules answer stands (*deferred*); LLM cascade is opt-in |
| 2 | **Chat routing** (`chat`) | LLM tool loop for every message (~2 rounds × system prompt + 10 tool schemas) | One Jev call; confident simple intents are answered by deterministic tool handlers with **no LLM** | `intent` (12 options incl. `prompt_injection`, `off_topic`, `action_request`, `complex_reasoning`), `spend_category`, `period`, `purchase_kind` Choices, `complexity` Score | LLM tool loop (cheap model if complexity is low) → regex router if no LLM |
| 3 | **Faithfulness of LLM text** (advisor, analyzer, forecaster, digest, chat) | Numeric grounding check only | Numeric check **and** a Jev claim-support check | `support` Choice: supported / unsupported_claim / contradiction / generic | numeric check alone (the previous behaviour) |
| 4 | **Goal type** (`forecaster`) | Every goal treated as savings that stay in the account | Spend goals (a trip, a laptop) leave the account at their deadline, and the balance bands include that outflow | `goal_type` Choice + `spend_at_deadline` Noul | keyword table |
| 5 | **Unfamiliar CSV columns** (`ingestion`) | Synonym list, otherwise the file was rejected | Jev reads header + sample values, and **code verifies** the values fit the role (dates parse, amounts are numbers) | `role` Choice | file rejected as before |
| 6 | **Subscription overlap for unknown services** (`analyzer`) | Lexicon groups only (Netflix + Hotstar) | Jev's `service_family` joins lexicon groups (Hoichoi + Netflix → overlapping video streaming) | from #1 | lexicon only |

What stays **out** of Jev, deliberately: arithmetic, dates, counting, thresholds and anything with an exact
answer. jev-1.13's own notes list math, date comparison and counting as weak spots. Amounts therefore arrive
pre-bucketed (`"small (under ₹500)"`), deadlines as `"in about 5 months"`, and money amounts in chat are parsed by
code.

## The confidence policy

A Jev answer is a distribution, so the code decides what to do with it ([`finpilot/decisions/policy.py`](../finpilot/decisions/policy.py)):

| Setting | Default | Meaning |
|---|---|---|
| `JEV_ACCEPT_CONFIDENCE` | 0.5 | Act on a Choice at/above this confidence; below it, the deterministic answer stands (*deferred*). |
| `JEV_ROUTE_CONFIDENCE` | 0.6 | Chat answered from tools (or refused) without an LLM at/above this. |
| `JEV_GUARD_CONFIDENCE` | 0.6 | LLM text is shown only if P(supported) ≥ this. |
| `JEV_ORDER_CHECK` | on | Answers with confidence in [accept, 0.9) are re-asked with the options **reversed** and the two distributions averaged. If the winner flips, the merged confidence collapses and the answer is deferred. |
| `CATEGORIZER_ESCALATE_TO_LLM` | off | Cascade deferred merchants to the LLM. |

Why the cascade is off by default: an independent benchmark (Towards Data Science, 3,080 support messages) found
that Jev's confidence did isolate the hard cases, but the fallback LLM solved only 57% of them. Every cascade it
tested did **worse** than Jev alone: it fixed 84 errors and introduced 211 new ones. Knowing a case is hard is not
the same as knowing another model can solve it. Enable the cascade only if `run_eval --jev` shows a gain on your
data.

Thresholds are data, not constants. The evaluation picks the category threshold **on a validation split**
(lowest threshold whose accepted answers reach 95% accuracy) and reports it **on a held-out test split**. Tuning
and scoring on the same data overstates every selective-prediction number.

Every decision lands in the `decisions` table: purpose, subject, answer, confidence, top probabilities, the action
taken (accepted / deferred / escalated / rejected) and the run it belongs to. It is browsable on the
**Decisions (Jev)** page and via `GET /api/decisions`.

## Cost — measured, estimated, and honest

Jev is billed at **$0.042 per 1M input tokens; output is free**. The table uses FinPilot's real request bodies,
measured at ~4 characters per token. The LLM column is the call each decision replaces, at gpt-4o-mini prices
($0.15 / $0.60 per 1M tokens).

| Decision | Input tokens / request | Jev $ / 1k decisions | LLM it replaces, $ / 1k |
|---|---|---|---|
| Chat routing | ~1,200 | $0.05 | **$0.67** (tool loop) |
| Merchant category | ~1,000 | $0.04 | $0.04 (batched) |
| Faithfulness check | ~230 | $0.01 | — new check |
| Goal type | ~340 | $0.01 | — new check |
| CSV column | ~220 | $0.01 | — new check |

* **Chat is where the money is.** A message Jev routes to a tool handler costs about 1/13 of the LLM tool loop it
  replaces, and those messages also skip 1–3 s of generation.
* **Categorization is roughly cost-neutral** against a *batched* cheap LLM, because the batch spreads its long
  prompt across 40 merchants. The reasons to switch are different:
  * answers are always a valid category (no parse failures, no invented labels);
  * confidence is calibrated, so "unsure" can be acted on;
  * each merchant is judged on its own, without being swayed by the other 39 in the batch;
  * latency is lower;
  * profiles are paid for once per merchant, ever.

  To keep the cost down, criteria carry `not_for` only where options are confused, and the service-family
  question is asked only for small debits.
* The guard, goal typing and CSV mapping are **new capabilities**, not savings, and the stats API reports them
  that way (`jev_cost_added_checks_usd`).

The live savings report (`GET /api/decisions/stats`, the **Decisions** page) counts only decisions that *replaced*
an LLM call. Chat answers that still escalated to the LLM are excluded.

## Engineering details

* **Client** ([`finpilot/decisions/jev.py`](../finpilot/decisions/jev.py)):
  * Client-side validation that mirrors the API's 422 rules: key pattern, ≥ 2 options, non-blank instructions, no
    criteria on Noul. A malformed question never costs a request.
  * Retries on 429/503/504 only (Mesh documents 500 as "do not retry").
  * The `X-Mesh-Version` header is pinned.
  * Structured criteria (`{what, not_for, examples}`) are sent as-is to TypeSafe's API, and flattened to strings for
    Mesh, whose contract documents string descriptions.
  * Attribution `tags` (`app`, `agent`, `purpose`) let Mesh usage reports split spend by decision.
* **Fan-out**: `evaluate_many` runs independent states in a thread pool. Each worker runs in a copy of the caller's
  `contextvars`, so every call is attributed to the right agent run and step.
* **Cache**: responses are cached by (model, state, questions) with option order included, so the order check
  never hits the cache. Repeated decisions are free, and cache hits are recorded with `cost_usd = 0`.
* **Telemetry**: Jev calls go to the same `llm_calls` table as LLM calls, with `kind = "jev"`, `purpose` and
  `cost_usd`. The coordinator reports them per step, separately from LLM calls.
* **Schema upgrade**: `init_db()` adds new nullable columns to databases created by earlier versions.
* **Fallbacks everywhere**: with no key, or Jev disabled, every agent behaves exactly as before.

## Evaluation

`python -m evaluation.run_eval --jev` measures every decision live against hand-labelled sets in
[`evaluation/jev_cases.py`](../evaluation/jev_cases.py):

* 142 merchant strings;
* 62 chat messages, including Hinglish, prompt injection, off-topic and "move my money" requests;
* 24 faithfulness pairs;
* 24 goal names;
* 20 CSV columns.

It reports accuracy, macro-F1, ECE with a reliability table, the threshold chosen on validation and scored on
test, order-flip rate, unsafe-message recall, false refusals, and cost and latency per 1k decisions.

The live run records `evaluation/cassettes/jev.json`; `--jev-replay` reproduces every number offline, in CI, with
no key.

Deterministic baselines are measured in every run, so the gap Jev has to close is visible up front:

| Decision | Deterministic path | Measured |
|---|---|---|
| Merchant category (142) | lexicon + keywords | 68.3% overall, **26.9%** on the 52 merchants it is unsure about (those go to Jev) |
| Chat routing (62) | regex router | 50.0% — and it cannot refuse the 12 unsafe/off-topic/action messages |
| Faithfulness (24) | numeric grounding check | catches **22%** of unfaithful drafts (14 of 18 contain no wrong number) |
| Goal type (24) | keyword table | 75% type, 92% spend-vs-keep |
| CSV columns (20) | synonym list | 11 headers unrecognised |

The Jev numbers are recorded only after a live run, and this repository never reports a model number it did not
measure. The development sandbox had no network access to `api.meshapi.ai`, so the code paths are tested against
a simulated `/v1/evaluate` (`tests/test_jev.py`, 25 tests).

## Sources

* Mesh API — [Evaluate](https://developers.meshapi.ai/docs/capabilities/evaluate) and the
  [`POST /v1/evaluate` reference](https://developers.meshapi.ai/api/evaluate/make-a-structured-decision)
  (request rules, `tags`, `X-Mesh-Version`, error codes, `typesafe/jev-1.13` pricing).
* TypeSafe — [Choice](https://docs.typesafe.ai/primitives/choice) (structured criteria, asking several questions
  per call), [Confidence](https://docs.typesafe.ai/confidence),
  [jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) (math, dates, counting, option order,
  large states), [Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions).
* Independent evaluation — [Jev vs. LLMs: When AI Moves from Generation to Decision-Making](https://towardsdatascience.com/jev-vs-llms-when-ai-moves-from-generation-to-decision-making/)
  (Towards Data Science): accuracy vs a local LLM, mid-band overconfidence, the failed cascade.
* Background — [Jev Explained](https://www.analyticsvidhya.com/blog/2026/09/jev-explained/) (Analytics Vidhya).
