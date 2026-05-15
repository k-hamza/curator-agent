"""
Base class for PDF-based collectors.

Provides PDF downloading and text extraction utilities.
Concrete collectors (ArxivCollector, etc.) inherit from this class.

BasePdfCollector is NOT registered in the collector registry —
only concrete subclasses are registered.
"""

import asyncio
from pathlib import Path

import httpx
import pdfplumber
from loguru import logger

from tech_watch.collectors.base import BaseCollector


class BasePdfCollector(BaseCollector):
    """
    Intermediate base class for collectors that download and extract PDF content.

    PDF-specific settings (timeout, max_chars, max_concurrent) are passed
    at construction time from AgentSettings — no hardcoded values here.

    Provides:
    - download_pdf()      : fetch a PDF from a URL and save it locally
    - extract_text()      : extract plain text from a local PDF file
    - fetch_and_extract() : download + extract in one step
    - fetch_many()        : download + extract multiple PDFs concurrently
    """

    HEADERS: dict[str, str] = {
        "User-Agent": (
            "tech-watch-agent/0.1 "
            "(research paper collector; +https://github.com/you/tech-watch-agent)"
        ),
    }

    def __init__(
        self,
        pdf_timeout: float = 30.0,
        pdf_max_chars: int = 8000,
        pdf_max_concurrent: int = 3,
    ) -> None:
        """
        Args:
            pdf_timeout:        HTTP timeout in seconds for PDF downloads.
            pdf_max_chars:      Maximum characters extracted per PDF.
            pdf_max_concurrent: Maximum PDFs downloaded in parallel.
        """
        self._timeout = httpx.Timeout(pdf_timeout, connect=5.0)
        self._max_chars = pdf_max_chars
        self._semaphore = asyncio.Semaphore(pdf_max_concurrent)

    # ---------------------------------------------------------------------------
    # PDF download
    # ---------------------------------------------------------------------------

    async def download_pdf(self, url: str, dest_path: Path) -> bool:
        """
        Download a PDF from url and save it to dest_path.

        Args:
            url:       URL of the PDF file.
            dest_path: Local path where the PDF will be saved.
                       Parent directory must exist.

        Returns:
            True on success, False on any error.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers=self.HEADERS,
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type and not url.endswith(".pdf"):
                    logger.warning(
                        f"Unexpected content-type '{content_type}' for PDF URL: {url}"
                    )

                dest_path.write_bytes(response.content)
                logger.debug(
                    f"Downloaded PDF ({len(response.content) / 1024:.1f} KB): "
                    f"{dest_path.name}"
                )
                return True

        except httpx.TimeoutException:
            logger.warning(f"Timeout downloading PDF: {url}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} downloading PDF: {url}")
        except httpx.RequestError as e:
            logger.warning(f"Network error downloading PDF {url}: {e}")
        except OSError as e:
            logger.error(f"Failed to write PDF to {dest_path}: {e}")

        return False

    # ---------------------------------------------------------------------------
    # Text extraction
    # ---------------------------------------------------------------------------

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract plain text from a local PDF file using pdfplumber.

        Extracts text page by page, concatenates with newlines,
        and truncates to _max_chars.

        Args:
            pdf_path: Path to a local PDF file.

        Returns:
            Extracted text string, possibly truncated. Empty string on failure.
        """
        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return ""

        try:
            extracted_pages: list[str] = []

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_pages.append(text.strip())

            full_text = "\n\n".join(extracted_pages)

            if not full_text.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path.name}")
                return ""

            if len(full_text) > self._max_chars:
                logger.debug(
                    f"Truncating PDF text from {len(full_text)} "
                    f"to {self._max_chars} chars: {pdf_path.name}"
                )
                full_text = full_text[: self._max_chars]

            logger.debug(
                f"Extracted {len(full_text)} chars from {pdf_path.name} "
                f"({len(extracted_pages)} pages)"
            )
            return full_text

        except Exception as e:
            logger.error(
                f"Failed to extract text from {pdf_path.name}: "
                f"{type(e).__name__}: {e}"
            )
            return ""

    # ---------------------------------------------------------------------------
    # Convenience methods
    # ---------------------------------------------------------------------------

    async def fetch_and_extract(self, url: str, dest_path: Path) -> str:
        """
        Download a PDF and extract its text in one step.

        Skips download if the file already exists on disk (simple caching).

        Args:
            url:       URL of the PDF to download.
            dest_path: Local path to save the PDF.

        Returns:
            Extracted text, or empty string on any failure.
        """
        if not dest_path.exists():
            success = await self.download_pdf(url, dest_path)
            if not success:
                return ""

        return self.extract_text(dest_path)

    async def fetch_many(
        self,
        items: list[tuple[str, Path]],
    ) -> list[str]:
        """
        Download and extract multiple PDFs concurrently.

        Respects the pdf_max_concurrent limit set at construction time.

        Args:
            items: List of (url, dest_path) tuples.

        Returns:
            List of extracted texts in the same order as items.
            Failed extractions produce an empty string at that index.
        """
        async def fetch_one(url: str, dest_path: Path) -> str:
            async with self._semaphore:
                return await self.fetch_and_extract(url, dest_path)

        return await asyncio.gather(
            *[fetch_one(url, path) for url, path in items]
        )
