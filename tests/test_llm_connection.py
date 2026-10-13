import httpx
import pytest
from openai import APIConnectionError

from finpilot.config import Settings
from finpilot.llm.client import LLMError, LLMRouter


def test_connection_error_identifies_endpoint_and_docker_fix(monkeypatch):
    router = LLMRouter(
        Settings(
            _env_file=None,
            llm_enabled=True,
            llm_provider="freellmapi",
            freellmapi_api_key="test-key",
            llm_cache_enabled=False,
        )
    )

    def unreachable(*args, **kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost:3001/v1/chat/completions"))

    monkeypatch.setattr(router, "_call", unreachable)
    with pytest.raises(LLMError, match=r"Cannot reach http://localhost:3001/v1.*host.docker.internal"):
        router.chat([{"role": "user", "content": "ping"}], agent="chat")
