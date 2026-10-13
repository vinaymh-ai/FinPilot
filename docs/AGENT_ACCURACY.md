# Agent reasoning and evidence

Categorization separates the payment channel from the economic purpose. Rules resolve known cases and user
corrections remain authoritative. Unresolved transactions reach the generative model with their full bank
narration, signed amount, date, currency and related observations. The prompt permits inference from the payee's
business and services, rather than requiring the bank to spell out a category. Output must use the taxonomy and
finite confidence values. Uncertain suggestions retain their reason and require review. Jev supplies a second
decision path where uncertainty remains.

Payments through an intermediary are grouped by narration and direction, not just its name. A card bill and
an online purchase through the same platform cannot inherit each other's category. Manual corrections remain
transaction-specific for payment intermediaries; generic provider names are not reused as merchant-wide purpose
rules. The classifier infers card repayments semantically; there is no rule declaring every CRED transaction a
card repayment.

Ask FinPilot treats greetings as conversation and answers them with the LLM without accessing the ledger.
For straightforward financial questions, code selects the merchant/category and dates, retrieves the matching
user's records, and calculates exact totals (including refunds). The LLM receives those results and a verified
summary, then explains the answer naturally. Complex questions use the financial tool loop. Historical chat text
is context, never transaction evidence. If a model replies without retrieving facts, it gets one retrieval reminder.

Numbers in financial replies must appear in supplied evidence. Jev checks whether the claims are supported;
when unavailable or when its preferred supported verdict misses the threshold, a separate LLM fact check runs.
Explicit unsupported/contradiction verdicts are rejected. Failed checks or upstream errors return a verified template.
Fallbacks are visible in route metadata. A legitimate zero-match result is never replaced by an overall total.

Merchant matching uses the payee and supported aliases, rather than arbitrary bank metadata. A full name in
a compact remark must share the truncated payee's first name. Thus a CRED lookup cannot include a Kiwi
payment merely because its creditor note says ICICI Cred. Overview periods use the current calendar and expose
their boundaries. Unverified balances are returned as null, and spending is distinguished from all bank debits.

The pipeline also tracks its input signature. Imported data, category corrections, balance changes or goal edits
invalidate previous results. Analysis excludes future-dated records and reports limited history. Forecasts require
a statement or user-confirmed balance, use the observed history length, and preserve multiple same-day simulated
shocks. Failures in critical stages stop dependent advice. Demo records do not contribute to a real imported ledger.

Regression coverage is in `tests/test_accuracy.py` and the existing agent, API and Jev tests. Tests exercise
payment intermediaries with different purposes, greeting behavior, exact merchant totals and periods, unsupported
model claims, user isolation, invalid confidences, manual overrides, stale forecasts and sparse histories.

Model inference is not proof of a payment's purpose. Confidence and review remain necessary when narrations
are incomplete. Likewise, forecasts express uncertainty rather than guaranteeing future balances.
