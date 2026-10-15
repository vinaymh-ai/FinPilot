# A 5-minute tour of FinPilot

This walkthrough follows the demo video: one person's bank statement goes in, and every agent's work is visible
in the UI. It works in zero-cost mode (`LLM_ENABLED=false`). With a `MESH_API_KEY`, the same steps also run the
LLM stages. To start the app, see [Quick start](../README.md#2-quick-start).

**Data.** Click **Try the demo**, load the *young professional* persona under **Settings → Demo data**, or upload
[`frontend/public/sample_transactions.csv`](../frontend/public/sample_transactions.csv) on **Transactions**.

## 1. Ingest and categorise — Coordinator, Categorizer
**Transactions → Upload statement.** The toast reports how many rows were imported, how many duplicates were
skipped, and that the agents ran. Re-uploading the same file adds nothing.
- The **How** column shows each label's source (`rule`, `keyword`, `p2p`, `memory`, `user`, `fallback`, and `llm`
  when a Mesh key is set) and its confidence. Hover it for the reason.
- The history icon on a row opens the category history: who set the label, when, and why.

## 2. Analyse behaviour — Analyzer
**Insights.** Patterns and anomalies come from rolling 30-day windows and robust outlier tests: category
spikes, overlapping subscriptions, possible duplicate charges, fees, unusually large purchases and price rises.
Subscriptions are detected from how regularly a charge repeats, not from a list of merchant names.

## 3. Forecast — Forecaster
**Overview.** *Where your money is heading* shows 400 Monte Carlo simulations of the balance, as a median line
and a P10–P90 band. **Goals** shows each goal's probability of being met on time. The hero card states this
week's key insight and one action.

## 4. Simulate a decision — What-if Lab
Click **Trip next month (₹45k)** (or build your own change). You get a verdict, how each goal's odds move, and
which spending to trim to pay for it. Baseline and scenario share the same random futures, so the difference
you see comes from the decision, not from noise.

## 5. Grounded advice — Advisor, Auditor
**Advisor.** Each recommendation carries a quantified impact and a concrete action. **Why?** opens its trigger,
evidence and reasoning. The Auditor checks every recommendation against this run's findings and data before it
is shown.

## 6. Ask FinPilot — chat team with memory
Try these in order in one conversation:
1. `How much did I spend on food delivery last month?`
2. `and in August?` — the same question with a different month (memory)
3. `which ones were above 500?` — lists the matching rows
4. `Can I afford a ₹45,000 trip next month?` — runs the same simulator as the What-if Lab
5. `mark Priya Nair as family`, then `yes` — creates a *Family* category and moves that payee's transactions only after you confirm
6. `summarise our chat`

Every figure in an answer is computed by code from your data and checked before it is shown. Expand an answer
to see the tools and SQL behind it.

## 7. Agents and decision trail
**Agents & Trail.** The LangGraph pipeline (plan → categorise → analyse → forecast → advise → audit) and the chat
team, with each run's step timeline and per-step details.

## 8. Proactive digests — Digest, APScheduler
**Digests → Send a digest now.** The digest arrives in the in-app inbox (and by email or Telegram when
configured). The schedule panel shows the next nightly re-analysis and the next weekly digest.

## 9. Self-evaluation
**Evaluation.** Shows categorisation accuracy, forecast backtests against naive baselines, chat answer accuracy
and the bank-narration benchmarks, including first-run scores. Reproduce them with `python -m evaluation.run_eval`
(or `make eval`). See also [AGENT_ACCURACY.md](AGENT_ACCURACY.md).
