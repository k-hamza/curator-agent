"""
Tests for the OpenAI-compatible LLM client.

HTTP calls are intercepted by patching _post() directly on the client instance.
"""

import json

import pytest
from pydantic import BaseModel

from tech_watch.llm.client import LLMClient, LLMError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client(
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen3:8b",
) -> LLMClient:
    """Build a minimal LLMClient for testing."""
    return LLMClient(base_url=base_url, model=model)


def make_completion_response(content: str) -> dict:
    """Build a minimal OpenAI-compatible chat completion response dict."""
    return {
        "id": "test-id",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Schema for complete_json tests
# ---------------------------------------------------------------------------

class ArticleScore(BaseModel):
    score: float
    topics: list[str]
    reasoning: str


# ---------------------------------------------------------------------------
# complete() tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_returns_text(monkeypatch) -> None:
    """complete() returns the assistant message content as plain text."""
    client = make_client()

    async def mock_post(payload: dict) -> dict:
        return make_completion_response("This is a summary.")

    monkeypatch.setattr(client, "_post", mock_post)

    result = await client.complete("Summarise this article.")
    assert result == "This is a summary."


@pytest.mark.asyncio
async def test_complete_strips_whitespace(monkeypatch) -> None:
    """complete() strips leading and trailing whitespace from the response."""
    client = make_client()

    async def mock_post(payload: dict) -> dict:
        return make_completion_response("  trimmed response  \n")

    monkeypatch.setattr(client, "_post", mock_post)

    result = await client.complete("prompt")
    assert result == "trimmed response"


@pytest.mark.asyncio
async def test_complete_raises_on_llm_error(monkeypatch) -> None:
    """complete() propagates LLMError raised by _post."""
    client = make_client()

    async def mock_post(payload: dict) -> dict:
        raise LLMError("HTTP 500: Internal server error")

    monkeypatch.setattr(client, "_post", mock_post)

    with pytest.raises(LLMError, match="HTTP 500"):
        await client.complete("prompt")


@pytest.mark.asyncio
async def test_complete_raises_on_malformed_response(monkeypatch) -> None:
    """complete() raises LLMError when the response structure is unexpected."""
    client = make_client()

    async def mock_post(payload: dict) -> dict:
        return {"unexpected": "structure"}

    monkeypatch.setattr(client, "_post", mock_post)

    with pytest.raises(LLMError, match="Unexpected LLM response structure"):
        await client.complete("prompt")


# ---------------------------------------------------------------------------
# complete_json() tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_json_parses_valid_response(monkeypatch) -> None:
    """complete_json() parses a valid JSON response into the schema model."""
    client = make_client()

    valid_json = json.dumps({
        "score": 0.85,
        "topics": ["LLM", "agents"],
        "reasoning": "Highly relevant to AI agents topic.",
    })

    async def mock_post(payload: dict) -> dict:
        return make_completion_response(valid_json)

    monkeypatch.setattr(client, "_post", mock_post)

    result = await client.complete_json("Score this article.", schema=ArticleScore)

    assert isinstance(result, ArticleScore)
    assert result.score == 0.85
    assert result.topics == ["LLM", "agents"]


@pytest.mark.asyncio
async def test_complete_json_strips_markdown_fences(monkeypatch) -> None:
    """complete_json() handles LLM responses wrapped in markdown code fences."""
    client = make_client()

    fenced_json = "```json\n" + json.dumps({
        "score": 0.7,
        "topics": ["DevOps"],
        "reasoning": "Relevant to infrastructure.",
    }) + "\n```"

    async def mock_post(payload: dict) -> dict:
        return make_completion_response(fenced_json)

    monkeypatch.setattr(client, "_post", mock_post)

    result = await client.complete_json("Score this.", schema=ArticleScore)
    assert result.score == 0.7
    assert result.topics == ["DevOps"]


@pytest.mark.asyncio
async def test_complete_json_raises_after_max_retries(monkeypatch) -> None:
    """complete_json() raises LLMError after exhausting all retries."""
    client = make_client()

    async def mock_post(payload: dict) -> dict:
        return make_completion_response("not valid json at all")

    monkeypatch.setattr(client, "_post", mock_post)

    with pytest.raises(LLMError, match="Failed to parse"):
        await client.complete_json(
            "Score this.", schema=ArticleScore, max_retries=1
        )


@pytest.mark.asyncio
async def test_complete_json_retries_on_invalid_json(monkeypatch) -> None:
    """complete_json() retries and succeeds after an initial invalid response."""
    client = make_client()

    call_count = 0
    valid_json = json.dumps({
        "score": 0.9,
        "topics": ["MLOps"],
        "reasoning": "Directly relevant.",
    })

    async def mock_post(payload: dict) -> dict:
        nonlocal call_count
        call_count += 1
        content = "oops not json" if call_count == 1 else valid_json
        return make_completion_response(content)

    monkeypatch.setattr(client, "_post", mock_post)

    result = await client.complete_json(
        "Score this.", schema=ArticleScore, max_retries=2
    )

    assert result.score == 0.9
    assert call_count == 2  # failed once, succeeded on retry
