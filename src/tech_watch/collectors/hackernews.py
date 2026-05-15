"""
HackerNews collector.

Uses the official Firebase HackerNews API (no authentication required).

API flow:
    GET /v0/topstories.json          → list of up to 500 item IDs
    GET /v0/item/{id}.json           → item detail (story, ask, job...)

Only "story" items with a URL are collected — Ask HN, polls, and job posts
are skipped because they don't point to external articles.

API reference: https://github.com/HackerNews/API
"""

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tech_watch.collectors.base_api import BaseApiCollector
from tech_watch.collectors.base import registry
from tech_watch.config.settings import BaseSourceSettings
from tech_watch.models.article import RawArticle, SourceType


# HackerNews Firebase API base URL
_HN_API_BASE = "https://hacker-news.firebaseio.com/v0"


@registry.register(SourceType.API)
class HackerNewsCollector(BaseApiCollector):
    """
    Collector for HackerNews top stories.

    Fetches the top stories list then retrieves each item concurrently.
    Only external-link stories are returned (type == 'story' and url present).
    """

    MAX_CONCURRENCY = 10  # HN Firebase API handles concurrent requests well

    async def collect(self, source: BaseSourceSettings) -> list[RawArticle]:
        """
        Fetch top HackerNews stories up to source.max_items.

        Args:
            source: Source configuration — max_items controls how many
                    stories are fetched after retrieving the top IDs list.

        Returns:
            List of RawArticle for each valid story found.
        """
        logger.info(f"[{source.name}] fetching HackerNews top stories")

        # Step 1 — fetch the ranked list of top story IDs
        top_ids = await self._fetch_top_ids()
        if not top_ids:
            return []

        # Limit early to avoid fetching more items than needed
        ids_to_fetch = top_ids[: source.max_items]
        logger.debug(f"[{source.name}] fetching {len(ids_to_fetch)} items")

        # Step 2 — fetch all items concurrently
        urls = [f"{_HN_API_BASE}/item/{id_}.json" for id_ in ids_to_fetch]
        raw_items = await self.get_many(urls)

        # Step 3 — convert valid items to RawArticle
        articles = []
        for item in raw_items:
            article = self._item_to_article(item, source)
            if article is not None:
                articles.append(article)

        logger.debug(
            f"[{source.name}] {len(articles)} valid stories "
            f"out of {len(ids_to_fetch)} fetched"
        )
        return articles

    async def _fetch_top_ids(self) -> list[int]:
        """
        Fetch the ranked list of top story IDs from HackerNews.
        Returns an empty list on failure.
        """
        data = await self.get_json(f"{_HN_API_BASE}/topstories.json")
        if not isinstance(data, list):
            logger.warning("HackerNews top stories response was not a list")
            return []
        return data

    def _item_to_article(
        self,
        item: Any,
        source: BaseSourceSettings,
    ) -> RawArticle | None:
        """
        Convert a raw HN item dict into a RawArticle.

        Skips:
        - None items (failed fetch)
        - Non-story items (Ask HN, polls, jobs)
        - Stories without an external URL (self-posts)
        - Dead or deleted items
        """
        if not isinstance(item, dict):
            return None

        # Skip dead or deleted items
        if item.get("dead") or item.get("deleted"):
            return None

        # Only collect stories that link to an external URL
        if item.get("type") != "story":
            return None

        url = item.get("url", "").strip()
        title = item.get("title", "").strip()

        if not url or not title:
            return None

        # Build content from title + optional text body (Ask HN style)
        # For standard stories the text field is empty
        text_body = item.get("text", "").strip()
        content = f"{title}\n\n{text_body}".strip() if text_body else title

        # HN timestamps are Unix epoch integers
        published_at: datetime | None = None
        timestamp = item.get("time")
        if isinstance(timestamp, int):
            published_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        # Author is the 'by' field
        author = item.get("by", "").strip()

        return RawArticle(
            url=url,
            title=title,
            source_type=SourceType.API,
            source_name=source.name,
            content=content,
            authors=[author] if author else [],
            published_at=published_at,
            external_id=str(item.get("id", "")),
        )
