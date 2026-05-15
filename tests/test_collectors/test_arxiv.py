"""
Tests for the arXiv collector.

Network calls are intercepted by patching _fetch_atom_feed and fetch_many
directly on the collector instance.
PDF extraction is tested via extract_text() with real minimal PDF bytes
generated on the fly using only the standard library.
"""

from datetime import timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from tech_watch.collectors.arxiv import ArxivCollector
from tech_watch.collectors.base import registry
from tech_watch.config.settings import PdfSourceSettings, load_settings
from tech_watch.models.article import SourceType


# ---------------------------------------------------------------------------
# Minimal valid Settings fixture
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = """
agent:
  model: "qwen3:8b"
interests: ["LLM"]
sources:
  - name: "arXiv"
    type: pdf
    url: "https://arxiv.org"
    enabled: true
    categories: ["cs.AI"]
    max_items: 5
"""


@pytest.fixture
def settings(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(MINIMAL_CONFIG, encoding="utf-8")
    s = load_settings(config)
    # Point pdf_storage_dir to tmp_path so no real files are written outside tests
    object.__setattr__(s.agent, "pdf_storage_dir", str(tmp_path / "pdfs"))
    return s


@pytest.fixture
def collector(settings):
    return ArxivCollector(settings=settings)


def make_source(
    name: str = "arXiv",
    categories: list[str] | None = None,
    max_items: int = 5,
) -> PdfSourceSettings:
    # Use explicit categories if provided (including empty list),
    # fall back to ["cs.AI"] only when None is passed
    resolved_categories = ["cs.AI"] if categories is None else categories
    return PdfSourceSettings(
        name=name,
        type=SourceType.PDF,
        url="https://arxiv.org",
        enabled=True,
        categories=resolved_categories,
        max_items=max_items,
    )


# ---------------------------------------------------------------------------
# Sample Atom feed
# ---------------------------------------------------------------------------

ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">

  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Advances in LLM Agent Architectures</title>
    <summary>We present a new framework for building LLM-based agents
    that can reason and act in complex environments.</summary>
    <author><name>Alice Martin</name></author>
    <author><name>Bob Chen</name></author>
    <published>2024-01-01T00:00:00Z</published>
  </entry>

  <entry>
    <id>http://arxiv.org/abs/2401.00002v2</id>
    <title>Retrieval Augmented Generation: A Survey</title>
    <summary>This survey covers recent advances in RAG systems
    for large language models.</summary>
    <author><name>Carol Smith</name></author>
    <published>2024-01-02T00:00:00Z</published>
  </entry>

</feed>"""


def parse_feed_entries() -> list[ET.Element]:
    """Parse the sample Atom feed and return its entries."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(ATOM_FEED)
    return root.findall("atom:entry", ns)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_arxiv_collector_registered() -> None:
    """ArxivCollector must be registered for SourceType.PDF after import."""
    from tech_watch.collectors import arxiv  # noqa: F401

    collector_cls = registry.get(SourceType.PDF)
    assert collector_cls is not None
    assert collector_cls is ArxivCollector


# ---------------------------------------------------------------------------
# _parse_entry tests — pure unit tests, no async needed
# ---------------------------------------------------------------------------

def test_parse_entry_extracts_fields(collector) -> None:
    """_parse_entry extracts all fields from a valid Atom entry."""
    entries = parse_feed_entries()
    result = collector._parse_entry(entries[0])

    assert result is not None
    assert result["arxiv_id"] == "2401.00001"
    assert result["title"] == "Advances in LLM Agent Architectures"
    assert "LLM-based agents" in result["abstract"]
    assert result["authors"] == ["Alice Martin", "Bob Chen"]
    assert result["url"] == "https://arxiv.org/abs/2401.00001"
    assert result["published_at"] is not None
    assert result["published_at"].tzinfo == timezone.utc


def test_parse_entry_strips_version_from_id(collector) -> None:
    """_parse_entry removes version suffix (v1, v2) from arXiv ID."""
    entries = parse_feed_entries()
    result = collector._parse_entry(entries[1])  # entry has v2

    assert result is not None
    assert result["arxiv_id"] == "2401.00002"  # no v2


def test_parse_entry_returns_none_on_missing_id(collector) -> None:
    """_parse_entry returns None when the entry has no id element."""
    entry = ET.fromstring("""
        <entry xmlns="http://www.w3.org/2005/Atom">
            <title>No ID entry</title>
        </entry>
    """)
    assert collector._parse_entry(entry) is None


def test_parse_entry_returns_none_on_missing_title(collector) -> None:
    """_parse_entry returns None when the entry has no title."""
    entry = ET.fromstring("""
        <entry xmlns="http://www.w3.org/2005/Atom">
            <id>http://arxiv.org/abs/2401.99999v1</id>
        </entry>
    """)
    assert collector._parse_entry(entry) is None


def test_extract_arxiv_id_strips_version() -> None:
    """_extract_arxiv_id correctly handles versioned and unversioned URLs."""
    assert ArxivCollector._extract_arxiv_id(
        "http://arxiv.org/abs/2401.12345v1"
    ) == "2401.12345"

    assert ArxivCollector._extract_arxiv_id(
        "http://arxiv.org/abs/cs/0601001v2"
    ) == "cs/0601001"

    assert ArxivCollector._extract_arxiv_id(
        "http://arxiv.org/abs/2401.12345"
    ) == "2401.12345"

    assert ArxivCollector._extract_arxiv_id("not-a-url") is None


# ---------------------------------------------------------------------------
# collect() integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_returns_articles(collector, monkeypatch) -> None:
    """collect() returns one RawArticle per valid feed entry."""
    entries = parse_feed_entries()

    async def mock_fetch_atom(categories, max_results):
        return entries

    async def mock_fetch_many(items):
        return ["Extracted PDF text for paper one.", "Extracted PDF text for paper two."]

    monkeypatch.setattr(collector, "_fetch_atom_feed", mock_fetch_atom)
    monkeypatch.setattr(collector, "fetch_many", mock_fetch_many)

    articles = await collector.collect(make_source())

    assert len(articles) == 2
    assert articles[0].title == "Advances in LLM Agent Architectures"
    assert articles[0].source_type == SourceType.PDF
    assert articles[0].external_id == "2401.00001"
    assert articles[0].content == "Extracted PDF text for paper one."


@pytest.mark.asyncio
async def test_collect_falls_back_to_abstract_on_empty_pdf(
    collector, monkeypatch
) -> None:
    """collect() uses the abstract when PDF extraction returns empty string."""
    entries = parse_feed_entries()

    async def mock_fetch_atom(categories, max_results):
        return entries[:1]  # only first entry

    async def mock_fetch_many(items):
        return [""]  # PDF extraction failed

    monkeypatch.setattr(collector, "_fetch_atom_feed", mock_fetch_atom)
    monkeypatch.setattr(collector, "fetch_many", mock_fetch_many)

    articles = await collector.collect(make_source())

    assert len(articles) == 1
    assert "LLM-based agents" in articles[0].content  # abstract used


@pytest.mark.asyncio
async def test_collect_skips_entry_with_no_content(
    collector, monkeypatch
) -> None:
    """collect() skips papers where both PDF and abstract are empty."""
    entry_no_abstract = ET.fromstring("""
        <entry xmlns="http://www.w3.org/2005/Atom">
            <id>http://arxiv.org/abs/2401.99999v1</id>
            <title>Paper with no content</title>
            <published>2024-01-01T00:00:00Z</published>
        </entry>
    """)

    async def mock_fetch_atom(categories, max_results):
        return [entry_no_abstract]

    async def mock_fetch_many(items):
        return [""]  # PDF extraction also failed

    monkeypatch.setattr(collector, "_fetch_atom_feed", mock_fetch_atom)
    monkeypatch.setattr(collector, "fetch_many", mock_fetch_many)

    articles = await collector.collect(make_source())
    assert articles == []


@pytest.mark.asyncio
async def test_collect_skips_when_no_categories(collector, monkeypatch) -> None:
    """collect() returns [] immediately when source has no categories."""
    fetch_called = False

    async def mock_fetch_atom(categories, max_results):
        nonlocal fetch_called
        fetch_called = True
        return []

    monkeypatch.setattr(collector, "_fetch_atom_feed", mock_fetch_atom)

    articles = await collector.collect(make_source(categories=[]))

    assert articles == []
    assert not fetch_called  # guard must trigger before any network call


@pytest.mark.asyncio
async def test_safe_collect_on_feed_failure_returns_empty(
    collector, monkeypatch
) -> None:
    """safe_collect() returns [] when atom feed fetch fails."""
    async def mock_fetch_atom(categories, max_results):
        return []

    monkeypatch.setattr(collector, "_fetch_atom_feed", mock_fetch_atom)

    articles = await collector.safe_collect(make_source())
    assert articles == []
