"""
Tests for the SummaryAgent.

LLM calls are intercepted by patching complete_json() on the LLMClient instance.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tech_watch.agents.summary import SummaryAgent, SummaryResponse
from tech_watch.config.settings import load_settings
from tech_watch.llm.client import LLMClient, LLMError
from tech_watch.models.article import ScoredArticle, SummarizedArticle, SourceType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = """
agent:
  model: "qwen3:8b"
  relevance_threshold: 0.6
interests:
  - "LLM"
sources:
  - name: "Test Feed"
    type: rss
    url: "https://example.com/feed.xml"
    enabled: true
"""


@pytest.fixture
def settings(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(MINIMAL_CONFIG, encoding="utf-8")
    return load_settings(config)


@pytest.fixture
def llm_client():
    return LLMClient(base_url="http://localhost:11434/v1", model="qwen3:8b")


@pytest.fixture
def agent(settings, llm_client):
    return SummaryAgent(settings=settings, llm_client=llm_client)


def make_scored_article(
    url: str = "https://example.com/article",
    title: str = "Advances in LLM Agents",
    content: str = "This article covers new approaches to LLM-based agent architectures.",
    source_type: SourceType = SourceType.RSS,
    relevance_score: float = 0.85,
) -> ScoredArticle:
    return ScoredArticle(
        url=url,
        title=title,
        source_type=source_type,
        source_name="Test Feed",
        content=content,
        relevance_score=relevance_score,
        matched_topics=["LLM"],
        filter_reasoning="Directly relevant.",
    )


def make_summary_response(
    summary: str = "A concise summary of the article.",
    key_points: list[str] | None = None,
) -> SummaryResponse:
    return SummaryResponse(
        summary=summary,
        key_points=key_points or [
            "Key point one",
            "Key point two",
            "Key point three",
        ],
    )


# ---------------------------------------------------------------------------
# Nominal tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_returns_summarized_articles(agent, llm_client) -> None:
    """summarize() returns SummarizedArticle for each article."""
    llm_client.complete_json = AsyncMock(
        return_value=make_summary_response()
    )

    articles = [make_scored_article()]
    result = await agent.summarize(articles)

    assert len(result) == 1
    assert isinstance(result[0], SummarizedArticle)
    assert result[0].summary == "A concise summary of the article."
    assert len(result[0].key_points) == 3


@pytest.mark.asyncio
async def test_summarize_preserves_scored_article_fields(
    agent, llm_client
) -> None:
    """SummarizedArticle retains all fields from the ScoredArticle."""
    llm_client.complete_json = AsyncMock(
        return_value=make_summary_response()
    )

    article = make_scored_article(
        url="https://example.com/specific",
        title="Specific Title",
        relevance_score=0.92,
    )
    result = await agent.summarize([article])

    assert result[0].url == "https://example.com/specific"
    assert result[0].title == "Specific Title"
    assert result[0].relevance_score == 0.92
    assert result[0].matched_topics == ["LLM"]


@pytest.mark.asyncio
async def test_summarize_multiple_articles(agent, llm_client) -> None:
    """summarize() processes all articles and returns results in order."""
    responses = [
        make_summary_response(summary="Summary one."),
        make_summary_response(summary="Summary two."),
        make_summary_response(summary="Summary three."),
    ]
    llm_client.complete_json = AsyncMock(side_effect=responses)

    articles = [
        make_scored_article(url=f"https://example.com/article-{i}")
        for i in range(3)
    ]
    result = await agent.summarize(articles)

    assert len(result) == 3
    assert result[0].summary == "Summary one."
    assert result[1].summary == "Summary two."
    assert result[2].summary == "Summary three."


@pytest.mark.asyncio
async def test_summarize_empty_input_returns_empty(agent) -> None:
    """summarize() returns [] immediately when given an empty list."""
    result = await agent.summarize([])
    assert result == []


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_drops_article_on_llm_error(agent, llm_client) -> None:
    """summarize() drops articles when LLM call fails."""
    llm_client.complete_json = AsyncMock(side_effect=LLMError("timeout"))

    articles = [make_scored_article()]
    result = await agent.summarize(articles)

    assert result == []


@pytest.mark.asyncio
async def test_summarize_continues_after_single_llm_error(
    agent, llm_client
) -> None:
    """summarize() continues processing remaining articles after a failure."""
    responses = [
        LLMError("timeout on first article"),
        make_summary_response(summary="Summary of second article."),
    ]
    llm_client.complete_json = AsyncMock(side_effect=responses)

    articles = [
        make_scored_article(url="https://example.com/fails"),
        make_scored_article(url="https://example.com/succeeds"),
    ]
    result = await agent.summarize(articles)

    assert len(result) == 1
    assert result[0].url == "https://example.com/succeeds"
    assert result[0].summary == "Summary of second article."
