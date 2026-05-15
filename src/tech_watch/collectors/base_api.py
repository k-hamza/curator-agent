"""
Base class for API-based collectors.

Provides a shared async HTTP client, JSON fetching utilities, and
rate-limiting helpers. Concrete collectors (HackerNews, etc.) inherit
from this class instead of BaseCollector directly.

BaseApiCollector is NOT registered in the collector registry —
only concrete subclasses are registered.
"""

import asyncio
from typing import Any

import httpx
from loguru import logger

from tech_watch.collectors.base import BaseCollector


# ---------------------------------------------------------------------------
# Base API collector
# ---------------------------------------------------------------------------

class BaseApiCollector(BaseCollector):
    """
    Intermediate base class for collectors that fetch JSON from HTTP APIs.

    Provides:
    - A pre-configured async HTTP client (_client_kwargs)
    - get_json()  : fetch a URL and return parsed JSON
    - get_many()  : fetch multiple URLs concurrently with a concurrency limit
    """

    # Default HTTP settings — subclasses can override as class attributes
    TIMEOUT: httpx.Timeout = httpx.Timeout(10.0, connect=5.0)
    HEADERS: dict[str, str] = {
        "User-Agent": (
            "tech-watch-agent/0.1 "
            "(tech watch bot; +https://github.com/you/tech-watch-agent)"
        ),
        "Accept": "application/json",
    }
    # Maximum number of concurrent requests when using get_many()
    MAX_CONCURRENCY: int = 5

    # ---------------------------------------------------------------------------
    # HTTP helpers
    # ---------------------------------------------------------------------------

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> Any | None:
        """
        Fetch a URL and return the parsed JSON response.

        Args:
            url:    Full URL to fetch.
            params: Optional query string parameters.

        Returns:
            Parsed JSON (dict, list, etc.) or None on any error.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUT,
                headers=self.HEADERS,
                follow_redirects=True,
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching: {url}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} fetching: {url}")
        except httpx.RequestError as e:
            logger.warning(f"Network error fetching {url}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error fetching {url}: {type(e).__name__}: {e}")

        return None

    async def get_many(
        self,
        urls: list[str],
        params: dict[str, Any] | None = None,
    ) -> list[Any | None]:
        """
        Fetch multiple URLs concurrently, respecting MAX_CONCURRENCY.

        Returns a list of parsed JSON responses in the same order as urls.
        Failed requests produce None at the corresponding index.

        Args:
            urls:   List of URLs to fetch.
            params: Query parameters applied to every request.

        Returns:
            List of JSON responses (or None for failures), same length as urls.
        """
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENCY)

        async def fetch_one(url: str) -> Any | None:
            async with semaphore:
                return await self.get_json(url, params=params)

        return await asyncio.gather(*[fetch_one(url) for url in urls])
