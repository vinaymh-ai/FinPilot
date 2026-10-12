"""Shared agent scaffolding."""

from __future__ import annotations

import json
import logging
from typing import Any

from finpilot.decisions.jev import JevClient, get_jev
from finpilot.llm.client import LLMError, LLMRouter, get_router

log = logging.getLogger(__name__)


class BaseAgent:
    """An agent = a job + its own system prompt + its own tools + a deterministic fallback.

    Two kinds of model, both optional:
    * **System One (Jev)** for closed-set decisions — which category, which intent, is this
      faithful. Cheap (input tokens only), calibrated, and its answer is always a valid option.
    * **System Two (LLM)** for open-ended generation — phrasing advice, multi-step reasoning.
    When neither is configured every agent completes its job with rules and statistics.
    """

    name: str = "agent"
    role: str = ""
    system_prompt: str = ""

    def __init__(
        self,
        router: LLMRouter | None = None,
        use_llm: bool = True,
        jev: JevClient | None = None,
        use_jev: bool = True,
    ):
        self.router = router or get_router()
        self.use_llm = use_llm
        self.jev = jev or get_jev()
        self.use_jev = use_jev

    @property
    def llm_available(self) -> bool:
        return self.use_llm and self.router.available(self.name)

    @property
    def jev_available(self) -> bool:
        return self.use_jev and self.jev.available()

    @property
    def model_label(self) -> str:
        return self.router.describe(self.name) if self.llm_available else "rules"

    def ask_json(
        self, user_payload: Any, *, max_tokens: int = 1500, temperature: float = 0.1
    ) -> tuple[Any, str]:
        """Send a JSON payload to the agent's own system prompt and parse a JSON reply."""
        content = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, default=str)
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}]
        data, resp = self.router.chat_json(
            messages, agent=self.name, max_tokens=max_tokens, temperature=temperature
        )
        return data, resp.label

    def safe_ask_json(self, user_payload: Any, **kwargs: Any) -> tuple[Any | None, str | None, str | None]:
        """Like ask_json but never raises: returns (data, model_label, error)."""
        if not self.llm_available:
            return None, None, "llm unavailable"
        try:
            data, label = self.ask_json(user_payload, **kwargs)
            return data, label, None
        except (LLMError, ValueError) as exc:
            log.warning("%s: LLM step failed, using deterministic fallback: %s", self.name, exc)
            return None, None, str(exc)[:300]
