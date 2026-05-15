"""
arXiv collector.

Uses the arXiv Atom API to fetch recent papers by category,
then downloads and extracts text from each PDF.

API flow:
    GET https://export.arxiv.org/api/query
        ?search_query=cat:cs.AI+OR+cat:cs.LG
        &sortBy=submittedDate
        &sortOrder=descending
        &max_results=10
    → Atom feed with paper metadata + abstracts

    For each paper:
        GET https://arxiv.org/pdf/{arxiv_id}
        → PDF binary

API reference: https://info.arxiv.org/help/api/user-manual.html
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
from loguru import logger

from tech_watch.collectors.base import registry
from tech_watch.collectors.base_pdf import BasePdfCollector
from tech_watch.config.settings import BaseSourceSettings, Settings
from tech_watch.models.article import RawArticle, SourceType


# arXiv Atom API endpoint
_ARXIV_API_URL = "https://export.arxiv.org/api/query"

# URL template for PDF download — filled with .format(arxiv_id=...)
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

# Atom XML namespaces used by arXiv
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@registry.register(SourceType.PDF)
class ArxivCollector(BasePdfCollector):
    """
    Collector for arXiv research papers.

    Fetches recent papers from configured categories via the Atom API,
    downloads each PDF, and extracts text for LLM summarization.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Application settings — provides all PDF config values
                      and the base storage directory.
        """
        super().__init__(
            pdf_timeout=settings.agent.pdf_timeout,
            pdf_max_chars=settings.agent.pdf_max_chars,
            pdf_max_concurrent=settings.agent.pdf_max_concurrent,
        )
        self._settings = settings
        self._http_timeout = httpx.Timeout(10.0, connect=5.0)
        self._headers = {
            "User-Agent": (
                "tech-watch-agent/0.1 "
                "(research paper collector; +https://github.com/you/tech-watch-agent)"
            )
        }

    def _pdf_dir_for_source(self, source_name: str) -> Path:
        """
        Return the PDF storage directory for a given source.
        Creates the directory if it doesn't exist.

        Path: {pdf_storage_dir}/{source_name}/
        e.g.: data/pdfs/arXiv/
        """
        safe_name = re.sub(r"[^\w\-]", "_", source_name)
        path = Path(self._settings.agent.pdf_storage_dir) / safe_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def collect(self, source: BaseSourceSettings) -> list[RawArticle]:
        """
        Fetch recent arXiv papers for the source's configured categories.

        Args:
            source: Source configuration.
                    source.categories — arXiv category codes (e.g. ['cs.AI', 'cs.LG'])
                    source.max_items  — maximum papers to fetch

        Returns:
            List of RawArticle with extracted PDF text as content.
            Falls back to abstract if PDF extraction fails.
        """
        categories = source.categories
        if not categories:
            logger.warning(f"[{source.name}] no categories configured — skipping")
            return []

        logger.info(
            f"[{source.name}] fetching arXiv papers "
            f"for categories: {categories}"
        )

        # Step 1 — fetch Atom feed with paper metadata
        entries = await self._fetch_atom_feed(categories, source.max_items)
        if not entries:
            return []

        logger.debug(f"[{source.name}] found {len(entries)} papers in feed")

        # Step 2 — prepare PDF download targets
        pdf_dir = self._pdf_dir_for_source(source.name)
        pdf_items: list[tuple[str, Path, dict]] = []

        for entry in entries:
            metadata = self._parse_entry(entry)
            if metadata is None:
                continue
            pdf_url = _ARXIV_PDF_URL.format(arxiv_id=metadata["arxiv_id"])
            safe_id = metadata["arxiv_id"].replace("/", "_")
            dest_path = pdf_dir / f"{safe_id}.pdf"
            pdf_items.append((pdf_url, dest_path, metadata))

        if not pdf_items:
            return []

        # Step 3 — download and extract PDFs concurrently
        texts = await self.fetch_many(
            [(url, path) for url, path, _ in pdf_items]
        )

        # Step 4 — assemble RawArticle for each paper
        articles = []
        for (pdf_url, dest_path, metadata), text in zip(pdf_items, texts):
            # Fall back to abstract if PDF extraction failed
            content = text if text.strip() else metadata.get("abstract", "")
            if not content:
                logger.warning(
                    f"[{source.name}] no content for paper "
                    f"{metadata['arxiv_id']} — skipping"
                )
                continue

            article = RawArticle(
                url=metadata["url"],
                title=metadata["title"],
                source_type=SourceType.PDF,
                source_name=source.name,
                content=content,
                authors=metadata.get("authors", []),
                published_at=metadata.get("published_at"),
                attachment_path=str(dest_path) if dest_path.exists() else None,
                external_id=metadata["arxiv_id"],
            )
            articles.append(article)

        logger.debug(f"[{source.name}] produced {len(articles)} articles")
        return articles

    # ---------------------------------------------------------------------------
    # Atom feed fetching
    # ---------------------------------------------------------------------------

    async def _fetch_atom_feed(
        self,
        categories: list[str],
        max_results: int,
    ) -> list[ET.Element]:
        """
        Query the arXiv API and return a list of Atom <entry> elements.

        Builds a search query combining all categories with OR,
        sorted by submission date descending.
        """
        search_query = " OR ".join(f"cat:{cat}" for cat in categories)

        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(max_results),
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._http_timeout,
                headers=self._headers,
                follow_redirects=True,
            ) as client:
                response = await client.get(_ARXIV_API_URL, params=params)
                response.raise_for_status()
                xml_content = response.text

        except httpx.TimeoutException:
            logger.warning("Timeout fetching arXiv Atom feed")
            return []
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} fetching arXiv feed")
            return []
        except httpx.RequestError as e:
            logger.warning(f"Network error fetching arXiv feed: {e}")
            return []

        try:
            root = ET.fromstring(xml_content)
            return root.findall("atom:entry", _NS)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv Atom feed: {e}")
            return []

    # ---------------------------------------------------------------------------
    # Entry parsing
    # ---------------------------------------------------------------------------

    def _parse_entry(self, entry: ET.Element) -> dict | None:
        """
        Parse an Atom <entry> element into a metadata dict.

        Returns None if essential fields (id, title) are missing.
        """
        id_element = entry.find("atom:id", _NS)
        if id_element is None or not id_element.text:
            return None

        arxiv_id = self._extract_arxiv_id(id_element.text)
        if not arxiv_id:
            return None

        title_element = entry.find("atom:title", _NS)
        if title_element is None or not title_element.text:
            return None

        title = " ".join(title_element.text.strip().split())

        summary_element = entry.find("atom:summary", _NS)
        abstract = summary_element.text.strip() if summary_element is not None else ""

        authors = []
        for author_el in entry.findall("atom:author", _NS):
            name_el = author_el.find("atom:name", _NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        published_at: datetime | None = None
        published_el = entry.find("atom:published", _NS)
        if published_el is not None and published_el.text:
            try:
                published_at = datetime.fromisoformat(
                    published_el.text.replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except ValueError:
                pass

        return {
            "arxiv_id": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published_at": published_at,
        }

    @staticmethod
    def _extract_arxiv_id(id_url: str) -> str | None:
        """
        Extract the arXiv ID from a full arXiv URL.

        Examples:
            http://arxiv.org/abs/2401.12345v1  → 2401.12345
            http://arxiv.org/abs/cs/0601001v1  → cs/0601001
        """
        match = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", id_url)
        if match:
            return match.group(1)
        return None
