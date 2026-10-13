FinPilot
Build a Proactive Financial Agent

Mesh API is the official sponsor and using it is mandatory. Every LLM/AI call in your project must route through Mesh API. A submission that does not use Mesh API is not a valid submission.
The Challenge
Build FinPilot, a multi-agent system that ingests a user's transactions, understands their behavior, forecasts where their money is heading, simulates "what-if" decisions against their real data, and proactively surfaces specific, personalized recommendations.

This is not another budgeting dashboard. Apps that categorize transactions and show spending charts already exist and are very good at it. Their common limitation is that they are reactive and backward-looking: they tell you what already happened and wait for you to ask. The hard, valuable part of this challenge is forward-looking reasoning: understanding a person's behavior well enough to predict what will happen, to answer "can I afford this?" before they commit, and to proactively intervene with a specific action, not a generic tip.

The Story (what you are building, end to end)
A user uploads or logs their transactions.

A categorization agent labels each transaction (rent, food, subscriptions, impulse buys) with logic whose output genuinely depends on the transaction. An analysis agent studies behavior over time and detects patterns and anomalies: "food delivery is up 40% this month," "six subscriptions, two of them unused." A forecasting step projects where the user is heading if current behavior continues. An advisor agent reasons over all of this and produces specific, grounded recommendations: "cancel these two subscriptions to save this amount," "at this rate you miss your savings goal by March." The user can also ask "can I afford a trip next month?" and the system simulates it against their real trajectory and answers. A background job sends a scheduled digest with the key insight and one action. Every recommendation records why it was made.

What You Will Build
1. Multi-agent orchestration (the backbone)

A coordinator that routes work to specialized agents and assembles the result.
At least three genuinely distinct agents (for example categorizer, analyzer, forecaster, advisor). Distinct means each has its own job and its own prompt or its own tools — not the same prompt behind different names.
2. Ingestion and real categorization

Transactions are imported (CSV upload or manual logging) and categorized by logic that genuinely depends on the transaction, then stored. Hardcoded or random categories do not count.
3. Behavior analysis and anomaly detection

The system aggregates behavior over time and detects real patterns and anomalies in the user's own data — not generic observations.
4. Forecasting and the simulation loop (the core of this track)

The system projects the user's financial trajectory, and can simulate a "what-if" decision (a planned purchase, a savings goal) against that trajectory and report the effect. This forward-looking reasoning is the heart of the track and is what separates a real system from a categorization wrapper.
5. Proactive, grounded advice

Recommendations are specific and tied to the user's detected patterns and forecast, not generic advice. A generic "save more money" tip scores nothing; "cancel these specific items" scores high.
6. An auditable decision trail

Every recommendation records why it was made — which pattern or forecast triggered it.
7. Self-evaluation (we reward this heavily)

Build test data and measure how accurate your categorization and forecasts are, with the numbers recorded in the repo. A team that genuinely built this almost always measured whether it works.
Highlighted Bonus (optional, makes you stand out)
Structured agent framework (e.g. LangGraph) with an explicit analyze → forecast → advise flow.
Goal tracking with the advice adapting as the user progresses or falls behind.
Conversational layer that answers free-form money questions from the user's real data.
Scheduled proactive delivery (email or Telegram) of a personalized digest, driven by a real background scheduler (Celery / APScheduler / cron), not a manual button.
Suggested Stack (use what you are comfortable with)
Backend: Python (Flask or FastAPI)
Agents: LangGraph, CrewAI, AutoGen, or your own orchestrator
LLM access (mandatory): all LLM/AI calls must go through Mesh API, the official sponsor. Mesh is an OpenAI-compatible gateway (base_url="https://api.meshapi.ai/v1", key prefix rsk_): one key gives access to many models. Point the OpenAI SDK at Mesh and read your key from the MESH_API_KEY environment variable. Full guide: Mesh API docs.
Database: SQLite or PostgreSQL
Frontend: a simple web UI or API (Streamlit, Gradio, Jinja2, or a JS frontend)
Scheduling (bonus): Celery, APScheduler, or cron
You may use synthetic or sample transaction data. Keep your API keys in a local, gitignored .env. Since your code is what is evaluated, your credentials are never needed by us.

Submission Requirements
A public GitHub repository with all your code and a README explaining what you built, how the agents cooperate, and how to run it.
A short note on your key design decisions and your self-evaluation results (your test data and the numbers).
(Optional) A short demo video and a deployed URL — reviewed for shortlisted submissions.
What a Great Submission Looks Like
Distinct cooperating agents with a coordinator, not one prompt doing everything.
Categorization and anomaly detection that genuinely respond to the user's data.
A real forecast or simulation you can demonstrate ("if you keep this up, here is what happens").
Advice grounded in the user's own patterns, a visible reason for each recommendation, and honest accuracy numbers.
Build something you would be proud to show in an interview.

Using Mesh API (required)
Mesh API is mandatory — every LLM/AI call must go through it. Mesh is an OpenAI-compatible gateway: one key gives access to many models (GPT, Claude, Gemini, and more).

Create an API key (it starts with rsk_) on the Mesh dashboard, then point the OpenAI SDK at Mesh and read your key from the environment:

import os
from openai import OpenAI

client = OpenAI(base_url="https://api.meshapi.ai/v1", api_key=os.getenv("MESH_API_KEY"))
client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
Add your key to a gitignored .env and as the GitHub secret MESH_API_KEY. Full guide: Mesh API docs.

Setup & Requirements
Follow this to make sure the automated checks pass and your project is eligible for evaluation.

Your repository must contain:

All your source code.
A requirements.txt (or pyproject.toml / Pipfile) listing a web framework (Flask or FastAPI) and the LLM client you use through Mesh API.
A README.md describing what you built, how your agents cooperate, your key design decisions, and your self-evaluation results.
A .gitignore that includes .env — never commit secrets.
Repository secrets (Settings → Secrets and variables → Actions) — new to GitHub secrets? Follow this short guide:

MESH_API_KEY — your Mesh API key. Mandatory; every LLM/AI call must go through Mesh.
SUBMISSION_TOKEN — your submission token, shown on your dashboard as soon as you register.
Your keys stay in your own repository and are only used by checks running in your own GitHub Actions — they are never sent to us.

Enabling the automated checks:

Download the workflow file: https://careerapi-production.krishnaik.in/api/ci/hackathons/agent-arena-2026/workflow.yml
Add it to your repository at .github/workflows/agent-arena-2026-checks.yml (create the .github/workflows/ folders if they don't exist).
Add the secrets listed above.
Push a commit. The checks run automatically — results appear in your repository's Actions tab and on your submission dashboard.
Critical checks (must pass for your commit to be eligible for evaluation):

Code compiles — all Python files are free of syntax errors.
Dependencies present — your requirements list a web framework and an LLM client.
Mesh API is used — your code routes LLM calls through Mesh (api.meshapi.ai base URL, or the meshapi SDK).
Mesh API key works — your MESH_API_KEY secret is a valid key. Verified with a metadata call that spends no credits.
Advisory checks (feedback only — these do not block you): no committed .env, a README is present, a .gitignore that ignores .env.

The automated checks are a first filter — passing them does not finalize your score. A failing critical check simply means "fix and push again"; it is not a penalty.

Mesh API — docs (mandatory LLM gateway)
LangGraph — agent orchestration
CrewAI
FastAPI
APScheduler — background jobs