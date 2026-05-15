"""
Tests for the RSS collector and collector registry.

HTTP requests are intercepted by patching RSSCollector._fetch_feed directly —
no real network calls are made.
"""

import hashlib
from datetime import timezone

import pytest

from tech_watch.collectors.base import autodiscover_collectors, registry
from tech_watch.collectors.rss import RSSCollector
from tech_watch.config.settings import RssSourceSettings
from tech_watch.models.article import SourceType


# ---------------------------------------------------------------------------
# Sample RSS feed fixtures
# ---------------------------------------------------------------------------

VALID_RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Tech Blog</title>
    <link>https://example.com</link>
    <description>A test feed</description>

    <item>
      <title>Introduction to LLM Agents</title>
      <link>https://example.com/llm-agents</link>
      <description>A deep dive into LLM-based agent architectures.</description>
      <author>Alice Martin</author>
      <pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate>
      <guid>https://example.com/llm-agents</guid>
    </item>

    <item>
      <title>DevOps Best Practices in 2024</title>
      <link>https://example.com/devops-2024</link>
      <description>How to streamline your CI/CD pipelines.</description>
      <pubDate>Tue, 02 Jan 2024 10:00:00 +0000</pubDate>
      <guid>https://example.com/devops-2024</guid>
    </item>

  </channel>
</rss>"""

EMPTY_RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
    <link>https://example.com</link>
    <description>No items here</description>
  </channel>
</rss>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_source(
    url: str = "https://example.com/feed.xml",
    name: str = "Test Feed",
    max_items: int = 20,
) -> RssSourceSettings:
    """Build a minimal RssSourceSettings for testing."""
    return RssSourceSettings(
        name=name,
        type=SourceType.RSS,
        url=url,
        enabled=True,
        max_items=max_items,
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_rss_collector_registered() -> None:
    """RSSCollector must be registered for SourceType.RSS after import."""
    from tech_watch.collectors import rss  # noqa: F401

    collector_cls = registry.get(SourceType.RSS)
    assert collector_cls is not None
    assert collector_cls is RSSCollector


def test_autodiscover_registers_rss() -> None:
    """autodiscover_collectors() must populate the registry with RSS at minimum."""
    autodiscover_collectors()
    assert SourceType.RSS in registry.registered_types()


# ---------------------------------------------------------------------------
# Nominal collection tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_returns_articles(monkeypatch) -> None:
    """collect() returns one RawArticle per feed entry."""
    collector = RSSCollector()

    async def mock_fetch(url: str) -> bytes:
        return VALID_RSS_FEED.encode("utf-8")

    monkeypatch.setattr(collector, "_fetch_feed", mock_fetch)

    articles = await collector.collect(make_source())

    assert len(articles) == 2
    assert articles[0].title == "Introduction to LLM Agents"
    assert articles[0].url == "https://example.com/llm-agents"
    assert articles[0].source_type == SourceType.RSS
    assert articles[0].source_name == "Test Feed"


@pytest.mark.asyncio
async def test_collect_respects_max_items(monkeypatch) -> None:
    """collect() returns at most max_items articles."""
    collector = RSSCollector()

    async def mock_fetch(url: str) -> bytes:
        return VALID_RSS_FEED.encode("utf-8")

    monkeypatch.setattr(collector, "_fetch_feed", mock_fetch)

    articles = await collector.collect(make_source(max_items=1))
    assert len(articles) == 1


@pytest.mark.asyncio
async def test_collect_parses_author(monkeypatch) -> None:
    """collect() extracts author name when present in the feed."""
    collector = RSSCollector()

    async def mock_fetch(url: str) -> bytes:
        return VALID_RSS_FEED.encode("utf-8")

    monkeypatch.setattr(collector, "_fetch_feed", mock_fetch)

    articles = await collector.collect(make_source())
    assert articles[0].authors == ["Alice Martin"]


@pytest.mark.asyncio
async def test_collect_parses_published_at(monkeypatch) -> None:
    """collect() parses publication date into a UTC-aware datetime."""
    collector = RSSCollector()

    async def mock_fetch(url: str) -> bytes:
        return VALID_RSS_FEED.encode("utf-8")

    monkeypatch.setattr(collector, "_fetch_feed", mock_fetch)

    articles = await collector.collect(make_source())
    assert articles[0].published_at is not None
    assert articles[0].published_at.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_collect_content_hash_is_deterministic(monkeypatch) -> None:
    """content_hash is stable — same URL always produces the same hash."""
    collector = RSSCollector()

    async def mock_fetch(url: str) -> bytes:
        return VALID_RSS_FEED.encode("utf-8")

    monkeypatch.setattr(collector, "_fetch_feed", mock_fetch)

    articles = await collector.collect(make_source())
    expected = hashlib.sha256(articles[0].url.encode()).hexdigest()
    assert articles[0].content_hash == expected


@pytest.mark.asyncio
async def test_collect_empty_feed_returns_empty_list(monkeypatch) -> None:
    """collect() returns an empty list when the feed has no entries."""
    collector = RSSCollector()

    async def mock_fetch(url: str) -> bytes:
        return EMPTY_RSS_FEED.encode("utf-8")

    monkeypatch.setattr(collector, "_fetch_feed", mock_fetch)

    articles = await collector.collect(make_source())
    assert articles == []


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safe_collect_on_fetch_failure_returns_empty_list(monkeypatch) -> None:
    """safe_collect() returns [] when _fetch_feed returns None (HTTP error)."""
    collector = RSSCollector()

    async def mock_fetch(url: str) -> None:
        return None

    monkeypatch.setattr(collector, "_fetch_feed", mock_fetch)

    articles = await collector.safe_collect(make_source())
    assert articles == []


@pytest.mark.asyncio
async def test_safe_collect_on_exception_returns_empty_list(monkeypatch) -> None:
    """safe_collect() returns [] when _fetch_feed raises an unexpected error."""
    collector = RSSCollector()

    async def mock_fetch(url: str) -> bytes:
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr(collector, "_fetch_feed", mock_fetch)

    articles = await collector.safe_collect(make_source())
    assert articles == []
