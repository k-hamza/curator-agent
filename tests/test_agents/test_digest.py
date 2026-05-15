"""
Tests for the DigestWriter.

LLM calls are intercepted by patching complete_json() on the LLMClient instance.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tech_watch.agents.digest import DigestWriter, OverviewResponse
from tech_watch.config.settings import load_settings
from tech_watch.llm.client import LLMClient, LLMError
from tech_watch.models.article import Digest, SourceType, SummarizedArticle


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
    return DigestWriter(settings=settings, llm_client=llm_client)


@pytest.fixture
def mock_overview(llm_client):
    """Patch LLM to return a fixed overview response."""
    llm_client.complete_json = AsyncMock(return_value=OverviewResponse(
        overview="Today's digest covers key advances in LLM agents and DevOps."
    ))


def make_summarized_article(
    url: str = "https://example.com/article",
    title: str = "Test Article",
    source_name: str = "Hugging Face Blog",
    source_type: SourceType = SourceType.RSS,
    relevance_score: float = 0.85,
    matched_topics: list[str] | None = None,
    summary: str = "A concise summary.",
    key_points: list[str] | None = None,
    published_at: datetime | None = None,
) -> SummarizedArticle:
    return SummarizedArticle(
        url=url,
        title=title,
        source_type=source_type,
        source_name=source_name,
        content="Full article content.",
        relevance_score=relevance_score,
        matched_topics=matched_topics or ["LLM"],
        filter_reasoning="Relevant.",
        summary=summary,
        key_points=key_points or ["Point one", "Point two", "Point three"],
        published_at=published_at,
    )


# ---------------------------------------------------------------------------
# Nominal tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_returns_digest(agent, mock_overview) -> None:
    """write() returns a Digest instance."""
    articles = [make_summarized_article()]
    result = await agent.write(articles, total_collected=10)

    assert isinstance(result, Digest)
    assert len(result.articles) == 1
    assert result.total_collected == 10
    assert result.total_filtered == 1


@pytest.mark.asyncio
async def test_write_includes_overview(agent, mock_overview) -> None:
    """write() includes the LLM-generated overview in the digest."""
    articles = [make_summarized_article()]
    result = await agent.write(articles)

    assert "LLM agents" in result.global_summary


@pytest.mark.asyncio
async def test_write_groups_by_source(agent, mock_overview) -> None:
    """write() groups articles by source_name."""
    articles = [
        make_summarized_article(
            url="https://example.com/a1",
            source_name="arXiv",
            source_type=SourceType.PDF,
        ),
        make_summarized_article(
            url="https://example.com/a2",
            source_name="HackerNews",
            source_type=SourceType.API,
        ),
        make_summarized_article(
            url="https://example.com/a3",
            source_name="arXiv",
            source_type=SourceType.PDF,
        ),
    ]
    result = await agent.write(articles)

    # Sources are sorted alphabetically: arXiv before HackerNews
    assert result.articles[0].source_name == "arXiv"
    assert result.articles[1].source_name == "arXiv"
    assert result.articles[2].source_name == "HackerNews"


@pytest.mark.asyncio
async def test_write_sorts_by_score_within_source(agent, mock_overview) -> None:
    """write() sorts articles by relevance score descending within each source."""
    articles = [
        make_summarized_article(
            url="https://example.com/low",
            source_name="arXiv",
            relevance_score=0.65,
        ),
        make_summarized_article(
            url="https://example.com/high",
            source_name="arXiv",
            relevance_score=0.95,
        ),
        make_summarized_article(
            url="https://example.com/mid",
            source_name="arXiv",
            relevance_score=0.80,
        ),
    ]
    result = await agent.write(articles)

    scores = [a.relevance_score for a in result.articles]
    assert scores == [0.95, 0.80, 0.65]


@pytest.mark.asyncio
async def test_write_sources_used_is_deduplicated(agent, mock_overview) -> None:
    """Digest.sources_used contains deduplicated source names."""
    articles = [
        make_summarized_article(url="https://example.com/a1", source_name="arXiv"),
        make_summarized_article(url="https://example.com/a2", source_name="arXiv"),
        make_summarized_article(url="https://example.com/a3", source_name="HackerNews"),
    ]
    result = await agent.write(articles)

    assert result.sources_used == ["HackerNews", "arXiv"]


@pytest.mark.asyncio
async def test_write_empty_input_returns_empty_digest(agent) -> None:
    """write() returns an empty Digest when no articles are provided."""
    result = await agent.write([], total_collected=25)

    assert isinstance(result, Digest)
    assert result.articles == []
    assert result.total_collected == 25
    assert result.total_filtered == 0
    assert result.global_summary == ""


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_overview_fallback_on_llm_error(agent, llm_client) -> None:
    """write() returns digest with empty overview when LLM fails."""
    llm_client.complete_json = AsyncMock(side_effect=LLMError("timeout"))

    articles = [make_summarized_article()]
    result = await agent.write(articles)

    assert isinstance(result, Digest)
    assert len(result.articles) == 1  # articles still present
    assert result.global_summary == ""  # overview empty but digest usable
