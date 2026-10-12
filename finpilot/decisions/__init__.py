"""System One decisions: closed-set judgments answered by Jev with calibrated probabilities.

LLMs generate text; Jev picks from options you define. FinPilot routes every *decision*
(which category, which intent, is this text faithful, what kind of goal) here, and keeps LLMs
for what only they can do — writing. See docs/JEV.md for the decision matrix.
"""

from finpilot.decisions.jev import (
    Choice,
    ChoiceAnswer,
    Evaluation,
    JevClient,
    JevError,
    JevRequestError,
    JevUnavailableError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    get_jev,
    set_jev,
)
from finpilot.decisions.policy import DecisionRecord, accepted, log_decisions, order_debias, record

__all__ = [
    "Choice",
    "ChoiceAnswer",
    "DecisionRecord",
    "Evaluation",
    "JevClient",
    "JevError",
    "JevRequestError",
    "JevUnavailableError",
    "Noul",
    "NoulAnswer",
    "Score",
    "ScoreAnswer",
    "accepted",
    "get_jev",
    "log_decisions",
    "order_debias",
    "record",
    "set_jev",
]
