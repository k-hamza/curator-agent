"""
Tests for the HackerNews collector.

All HTTP calls are intercepted by patching get_json() and get_many()
on the collector instance — no real network calls are made.
"""

import pytest
from datetime import timezone

from tech_watch.collectors.base import registry
from tech_watch.collectors.hackernews import HackerNewsCollector
from tech_watch.config.settings import ApiSourceSettings
from tech_watch.models.article import SourceType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Realistic HN API responses
TOP_IDS = [1001, 1002, 1003]

HN_ITEMS = {
    1001: {
        "id": 1001,
        "type": "story",
        "title": "LLM Agents are changing software development",
        "url": "https://example.com/llm-agents",
        "by": "alice",
        "time": 1704067200,  # 2024-01-01 00:00:00 UTC
        "score": 250,
    },
    1002: {
        "id": 1002,
        "type": "story",
        "title": "New approach to RAG pipelines",
        "url": "https://example.com/rag-pipelines",
        "by": "bob",
        "time": 1704153600,  # 2024-01-02 00:00:00 UTC
        "score": 180,
    },
    1003: {
        "id": 1003,
        "type": "ask",   # Ask HN — no external URL, should be skipped
        "title": "Ask HN: Best resources for learning ML?",
        "by": "charlie",
        "time": 1704240000,
        "score": 95,
    },
}


def make_source(
    name: str = "HackerNews",
    max_items: int = 20,
) -> ApiSourceSettings:
    """Build a minimal ApiSourceSettings for testing."""
    return ApiSourceSettings(
        name=name,
        type=SourceType.API,
        url="https://hacker-news.firebaseio.com",
        enabled=True,
        max_items=max_items,
    )


def make_items_list(ids: list[int]) -> list[dict]:
    """Return item dicts for the given IDs, simulating get_many() response."""
    return [HN_ITEMS.get(id_) for id_ in ids]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_hackernews_collector_registered() -> None:
    """HackerNewsCollector must be registered for SourceType.API after import."""
    from tech_watch.collectors import hackernews  # noqa: F401

    collector_cls = registry.get(SourceType.API)
    assert collector_cls is not None
    assert collector_cls is HackerNewsCollector


# ---------------------------------------------------------------------------
# Nominal collection tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_returns_stories(monkeypatch) -> None:
    """collect() returns one RawArticle per valid story."""
    collector = HackerNewsCollector()

    async def mock_fetch_top_ids() -> list[int]:
        return TOP_IDS

    async def mock_get_many(urls: list[str], **kwargs) -> list:
        return make_items_list(TOP_IDS)

    monkeypatch.setattr(collector, "_fetch_top_ids", mock_fetch_top_ids)
    monkeypatch.setattr(collector, "get_many", mock_get_many)

    articles = await collector.collect(make_source())

    # item 1003 is an Ask HN (no url) — should be skipped
    assert len(articles) == 2
    assert articles[0].title == "LLM Agents are changing software development"
    assert articles[0].url == "https://example.com/llm-agents"
    assert articles[0].source_type == SourceType.API
    assert articles[0].source_name == "HackerNews"


@pytest.mark.asyncio
async def test_collect_respects_max_items(monkeypatch) -> None:
    """collect() fetches at most max_items IDs."""
    collector = HackerNewsCollector()

    fetched_urls: list[str] = []

    async def mock_fetch_top_ids() -> list[int]:
        return TOP_IDS

    async def mock_get_many(urls: list[str], **kwargs) -> list:
        fetched_urls.extend(urls)
        return make_items_list([1001])  # only first item

    monkeypatch.setattr(collector, "_fetch_top_ids", mock_fetch_top_ids)
    monkeypatch.setattr(collector, "get_many", mock_get_many)

    await collector.collect(make_source(max_items=1))

    # Only 1 URL should have been fetched
    assert len(fetched_urls) == 1


@pytest.mark.asyncio
async def test_collect_parses_author_and_date(monkeypatch) -> None:
    """collect() extracts author and UTC datetime correctly."""
    collector = HackerNewsCollector()

    async def mock_fetch_top_ids() -> list[int]:
        return [1001]

    async def mock_get_many(urls: list[str], **kwargs) -> list:
        return [HN_ITEMS[1001]]

    monkeypatch.setattr(collector, "_fetch_top_ids", mock_fetch_top_ids)
    monkeypatch.setattr(collector, "get_many", mock_get_many)

    articles = await collector.collect(make_source())

    assert articles[0].authors == ["alice"]
    assert articles[0].published_at is not None
    assert articles[0].published_at.tzinfo == timezone.utc
    assert articles[0].published_at.year == 2024


@pytest.mark.asyncio
async def test_collect_stores_external_id(monkeypatch) -> None:
    """collect() stores the HN item ID as external_id."""
    collector = HackerNewsCollector()

    async def mock_fetch_top_ids() -> list[int]:
        return [1001]

    async def mock_get_many(urls: list[str], **kwargs) -> list:
        return [HN_ITEMS[1001]]

    monkeypatch.setattr(collector, "_fetch_top_ids", mock_fetch_top_ids)
    monkeypatch.setattr(collector, "get_many", mock_get_many)

    articles = await collector.collect(make_source())
    assert articles[0].external_id == "1001"


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_skips_ask_hn(monkeypatch) -> None:
    """collect() skips Ask HN items (type != 'story' or no url)."""
    collector = HackerNewsCollector()

    async def mock_fetch_top_ids() -> list[int]:
        return [1003]  # Ask HN item

    async def mock_get_many(urls: list[str], **kwargs) -> list:
        return [HN_ITEMS[1003]]

    monkeypatch.setattr(collector, "_fetch_top_ids", mock_fetch_top_ids)
    monkeypatch.setattr(collector, "get_many", mock_get_many)

    articles = await collector.collect(make_source())
    assert articles == []


@pytest.mark.asyncio
async def test_collect_skips_dead_items(monkeypatch) -> None:
    """collect() skips dead or deleted items."""
    collector = HackerNewsCollector()

    dead_item = {
        "id": 9999,
        "type": "story",
        "title": "This story is dead",
        "url": "https://example.com/dead",
        "by": "user",
        "time": 1704067200,
        "dead": True,
    }

    async def mock_fetch_top_ids() -> list[int]:
        return [9999]

    async def mock_get_many(urls: list[str], **kwargs) -> list:
        return [dead_item]

    monkeypatch.setattr(collector, "_fetch_top_ids", mock_fetch_top_ids)
    monkeypatch.setattr(collector, "get_many", mock_get_many)

    articles = await collector.collect(make_source())
    assert articles == []


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safe_collect_on_empty_top_ids_returns_empty_list(monkeypatch) -> None:
    """safe_collect() returns [] when top IDs fetch fails."""
    collector = HackerNewsCollector()

    async def mock_fetch_top_ids() -> list[int]:
        return []

    monkeypatch.setattr(collector, "_fetch_top_ids", mock_fetch_top_ids)

    articles = await collector.safe_collect(make_source())
    assert articles == []


@pytest.mark.asyncio
async def test_safe_collect_handles_none_items(monkeypatch) -> None:
    """safe_collect() skips None responses from failed item fetches."""
    collector = HackerNewsCollector()

    async def mock_fetch_top_ids() -> list[int]:
        return [1001, 9999]  # 9999 will return None (failed fetch)

    async def mock_get_many(urls: list[str], **kwargs) -> list:
        return [HN_ITEMS[1001], None]  # one success, one failure

    monkeypatch.setattr(collector, "_fetch_top_ids", mock_fetch_top_ids)
    monkeypatch.setattr(collector, "get_many", mock_get_many)

    articles = await collector.safe_collect(make_source())
    assert len(articles) == 1
    assert articles[0].title == "LLM Agents are changing software development"
