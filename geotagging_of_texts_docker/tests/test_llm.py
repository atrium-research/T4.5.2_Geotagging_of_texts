import httpx
import openai
import pytest

from app.llm import llama


def test_context_overflow_tells_the_user_to_raise_llama_context(monkeypatch):
    response = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    error = openai.BadRequestError(
        "request (7880 tokens) exceeds the available context size (4096 tokens), try increasing it",
        response=response, body=None)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(llama, "_with_connection_retry", fail)
    with pytest.raises(RuntimeError, match="LLAMA_CONTEXT"):
        llama.chat("m", [])
