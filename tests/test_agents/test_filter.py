"""
Tests for the FilterAgent.

LLM calls are intercepted by patching complete_json() on the LLMClient instance.
"""

import pytest
from unittest.mock import AsyncMock

from tech_watch.agents.filter import FilterAgent, FilterResponse
from tech_watch.config.settings import load_settings
from tech_watch.llm.client import LLMClient, LLMError
from tech_watch.models.article import RawArticle, ScoredArticle, SourceType
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = """
agent:
  model: "qwen3:8b"
  relevance_threshold: 0.6
interests:
  - "AI agents"
  - "LLM"
  - "DevOps"
  - "robotics"
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
    """Return a real LLMClient instance — its methods will be patched per test."""
    return LLMClient(base_url="http://localhost:11434/v1", model="qwen3:8b")


@pytest.fixture
def agent(settings, llm_client):
    return FilterAgent(settings=settings, llm_client=llm_client)


def make_article(
    url: str = "https://example.com/article",
    title: str = "Test Article",
    content: str = "Content about LLM agents and architectures.",
    source_type: SourceType = SourceType.RSS,
) -> RawArticle:
    return RawArticle(
        url=url,
        title=title,
        source_type=source_type,
        source_name="Test Feed",
        content=content,
    )


# ---------------------------------------------------------------------------
# Nominal tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filter_returns_scored_articles_above_threshold(
    agent, llm_client
) -> None:
    """filter() returns ScoredArticle for articles above the threshold."""
    llm_client.complete_json = AsyncMock(return_value=FilterResponse(
        score=0.9,
        matched_topics=["LLM", "AI agents"],
        reasoning="Directly relevant to LLM agents.",
    ))

    articles = [make_article()]
    result = await agent.filter(articles)

    assert len(result) == 1
    assert isinstance(result[0], ScoredArticle)
    assert result[0].relevance_score == 0.9
    assert result[0].matched_topics == ["LLM", "AI agents"]


@pytest.mark.asyncio
async def test_filter_discards_articles_below_threshold(
    agent, llm_client
) -> None:
    """filter() discards articles scoring below relevance_threshold (0.6)."""
    llm_client.complete_json = AsyncMock(return_value=FilterResponse(
        score=0.3,
        matched_topics=[],
        reasoning="Not relevant to any interest.",
    ))

    articles = [make_article(title="Tour de France 2024")]
    result = await agent.filter(articles)

    assert result == []


@pytest.mark.asyncio
async def test_filter_keeps_article_matching_single_interest(
    agent, llm_client
) -> None:
    """filter() keeps articles that match only one interest (OR logic)."""
    llm_client.complete_json = AsyncMock(return_value=FilterResponse(
        score=0.88,
        matched_topics=["robotics"],
        reasoning="Focused entirely on robotics control systems.",
    ))

    articles = [make_article(title="Advances in Robotic Control Systems")]
    result = await agent.filter(articles)

    assert len(result) == 1
    assert result[0].matched_topics == ["robotics"]
    assert result[0].relevance_score == 0.88


@pytest.mark.asyncio
async def test_filter_processes_multiple_articles(agent, llm_client) -> None:
    """filter() processes all articles and returns only those above threshold."""
    responses = [
        FilterResponse(score=0.9, matched_topics=["LLM"], reasoning="Relevant."),
        FilterResponse(score=0.2, matched_topics=[], reasoning="Not relevant."),
        FilterResponse(score=0.75, matched_topics=["DevOps"], reasoning="Relevant."),
    ]
    llm_client.complete_json = AsyncMock(side_effect=responses)

    articles = [
        make_article(url=f"https://example.com/article-{i}")
        for i in range(3)
    ]
    result = await agent.filter(articles)

    assert len(result) == 2
    assert result[0].relevance_score == 0.9
    assert result[1].relevance_score == 0.75


@pytest.mark.asyncio
async def test_filter_article_at_exact_threshold_is_kept(
    agent, llm_client
) -> None:
    """filter() keeps articles scoring exactly at the threshold (>=, not >)."""
    llm_client.complete_json = AsyncMock(return_value=FilterResponse(
        score=0.6,  # exactly at threshold
        matched_topics=["DevOps"],
        reasoning="Marginally relevant.",
    ))

    articles = [make_article()]
    result = await agent.filter(articles)

    assert len(result) == 1


@pytest.mark.asyncio
async def test_filter_preserves_raw_article_fields(agent, llm_client) -> None:
    """ScoredArticle retains all fields from the original RawArticle."""
    llm_client.complete_json = AsyncMock(return_value=FilterResponse(
        score=0.85,
        matched_topics=["LLM"],
        reasoning="Relevant.",
    ))

    article = make_article(
        url="https://example.com/specific",
        title="Specific Title",
        content="Specific content about LLMs.",
        source_type=SourceType.API,
    )
    result = await agent.filter([article])

    assert result[0].url == "https://example.com/specific"
    assert result[0].title == "Specific Title"
    assert result[0].source_type == SourceType.API


@pytest.mark.asyncio
async def test_filter_empty_input_returns_empty(agent) -> None:
    """filter() returns [] immediately when given an empty list."""
    result = await agent.filter([])
    assert result == []


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filter_discards_article_on_llm_error(agent, llm_client) -> None:
    """filter() silently discards articles when LLM call fails."""
    llm_client.complete_json = AsyncMock(side_effect=LLMError("timeout"))

    articles = [make_article()]
    result = await agent.filter(articles)

    assert result == []


@pytest.mark.asyncio
async def test_filter_continues_after_single_llm_error(
    agent, llm_client
) -> None:
    """filter() continues processing remaining articles after a single LLM failure."""
    responses = [
        LLMError("timeout on first article"),
        FilterResponse(score=0.8, matched_topics=["LLM"], reasoning="Relevant."),
    ]
    llm_client.complete_json = AsyncMock(side_effect=responses)

    articles = [
        make_article(url="https://example.com/fails"),
        make_article(url="https://example.com/succeeds"),
    ]
    result = await agent.filter(articles)

    assert len(result) == 1
    assert result[0].url == "https://example.com/succeeds"
