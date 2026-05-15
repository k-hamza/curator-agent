"""
RSS/Atom feed collector.

Fetches articles from RSS and Atom feeds using feedparser.
Registered for SourceType.RSS — handles any source configured with type: rss.

Design notes:
- One HTTP request per feed URL
- feedparser handles both RSS 2.0 and Atom formats transparently
- Content is extracted from multiple possible fields (content, summary, description)
  because RSS feeds are inconsistent in how they store article text
- published_at falls back to fetched_at if the feed does not provide a date
"""

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx
from loguru import logger

from tech_watch.collectors.base import BaseCollector, registry
from tech_watch.config.settings import BaseSourceSettings
from tech_watch.models.article import RawArticle, SourceType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(entry: feedparser.FeedParserDict) -> datetime | None:
    """
    Extract publication date from a feed entry.

    feedparser normalises dates into a time.struct_time via 'published_parsed'
    or 'updated_parsed'. We convert to a timezone-aware UTC datetime.
    Returns None if no date is available.
    """
    for field in ("published_parsed", "updated_parsed"):
        value = getattr(entry, field, None) or entry.get(field)
        if value is not None:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue

    # Some feeds only provide a raw RFC-2822 string
    for field in ("published", "updated"):
        raw = entry.get(field)
        if raw:
            try:
                return parsedate_to_datetime(raw).astimezone(timezone.utc)
            except Exception:
                continue

    return None


def _extract_content(entry: feedparser.FeedParserDict) -> str:
    """
    Extract the best available text content from a feed entry.

    Priority:
    1. entry.content[0].value  — full article body (Atom feeds)
    2. entry.summary           — article summary / description
    3. entry.description       — fallback, older RSS feeds
    4. entry.title             — last resort, never empty for valid entries
    """
    # Atom full content
    content_list = entry.get("content")
    if content_list and isinstance(content_list, list):
        value = content_list[0].get("value", "").strip()
        if value:
            return value

    # RSS summary / description
    for field in ("summary", "description"):
        value = entry.get(field, "").strip()
        if value:
            return value

    # Last resort
    return entry.get("title", "").strip()


def _extract_authors(entry: feedparser.FeedParserDict) -> list[str]:
    """Extract author names from a feed entry."""
    # Atom: list of author dicts with 'name' key
    authors = entry.get("authors", [])
    if authors:
        return [a.get("name", "").strip() for a in authors if a.get("name")]

    # RSS: single author string
    author = entry.get("author", "").strip()
    if author:
        return [author]

    return []


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

@registry.register(SourceType.RSS)
class RSSCollector(BaseCollector):
    """
    Collector for RSS and Atom feeds.

    Fetches the feed XML via HTTP, parses it with feedparser,
    and converts each entry into a RawArticle.
    """

    # feedparser can parse from a string, so we fetch the raw bytes ourselves
    # to control timeout, headers, and error handling
    _TIMEOUT = httpx.Timeout(10.0, connect=5.0)
    _HEADERS = {
        "User-Agent": "tech-watch-agent/0.1 (RSS reader; +https://github.com/you/tech-watch-agent)"
    }

    async def collect(self, source: BaseSourceSettings) -> list[RawArticle]:
        """
        Fetch and parse a single RSS/Atom feed.

        Args:
            source: Source configuration — url is the feed URL.

        Returns:
            List of RawArticle, capped at source.max_items.
        """
        logger.info(f"[{source.name}] fetching RSS feed: {source.url}")

        raw_xml = await self._fetch_feed(source.url)
        if raw_xml is None:
            return []

        feed = feedparser.parse(raw_xml)

        if feed.bozo and not feed.entries:
            # bozo=True means feedparser detected a malformed feed
            # but it often recovers partial content — only fail if no entries
            logger.warning(
                f"[{source.name}] malformed feed (bozo exception: {feed.bozo_exception})"
            )
            return []

        articles = []
        for entry in feed.entries[: source.max_items]:
            article = self._entry_to_article(entry, source)
            if article is not None:
                articles.append(article)

        logger.debug(f"[{source.name}] parsed {len(articles)} entries from feed")
        return articles

    async def _fetch_feed(self, url: str) -> bytes | None:
        """
        Fetch the raw feed content over HTTP.
        Returns None on any HTTP or network error.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self._TIMEOUT,
                headers=self._HEADERS,
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching feed: {url}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} fetching feed: {url}")
        except httpx.RequestError as e:
            logger.warning(f"Network error fetching feed {url}: {e}")
        return None

    def _entry_to_article(
        self,
        entry: feedparser.FeedParserDict,
        source: BaseSourceSettings,
    ) -> RawArticle | None:
        """
        Convert a feedparser entry dict into a RawArticle.
        Returns None if the entry lacks a URL or title (not a valid article).
        """
        url = entry.get("link", "").strip()
        title = entry.get("title", "").strip()

        if not url or not title:
            logger.debug(f"[{source.name}] skipping entry without url or title")
            return None

        return RawArticle(
            url=url,
            title=title,
            source_type=SourceType.RSS,
            source_name=source.name,
            content=_extract_content(entry),
            authors=_extract_authors(entry),
            published_at=_parse_date(entry),
            external_id=entry.get("id", "").strip() or None,
        )
