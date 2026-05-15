"""
Tests for the SQLite memory store.

Each test gets a fresh in-memory SQLite database via the store fixture —
no files written to disk, no state shared between tests.
"""

from datetime import timezone

import pytest

from tech_watch.memory.store import MemoryStore
from tech_watch.models.article import RawArticle, SourceType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> MemoryStore:
    """
    Fresh in-memory SQLite store for each test.
    ':memory:' creates a temporary database that disappears after the test.
    """
    s = MemoryStore(db_path=":memory:")
    s.init()
    return s


def make_article(
    url: str = "https://example.com/article-1",
    title: str = "Test Article",
    source_name: str = "Test Source",
    source_type: SourceType = SourceType.RSS,
) -> RawArticle:
    """Build a minimal RawArticle for testing."""
    return RawArticle(
        url=url,
        title=title,
        source_type=source_type,
        source_name=source_name,
        content="Some content about LLMs and agents.",
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_store_initialises_empty(store: MemoryStore) -> None:
    """A freshly initialised store contains no articles."""
    assert store.count() == 0


def test_init_is_idempotent(store: MemoryStore) -> None:
    """Calling init() multiple times does not raise or duplicate tables."""
    store.init()
    store.init()
    assert store.count() == 0


# ---------------------------------------------------------------------------
# is_seen
# ---------------------------------------------------------------------------

def test_is_seen_returns_false_for_unknown_article(store: MemoryStore) -> None:
    """is_seen() returns False for an article not in the store."""
    article = make_article()
    assert store.is_seen(article.content_hash) is False


def test_is_seen_returns_true_after_mark_seen(store: MemoryStore) -> None:
    """is_seen() returns True after the article has been marked as seen."""
    article = make_article()
    store.mark_seen([article])
    assert store.is_seen(article.content_hash) is True


# ---------------------------------------------------------------------------
# mark_seen
# ---------------------------------------------------------------------------

def test_mark_seen_increments_count(store: MemoryStore) -> None:
    """mark_seen() adds articles to the store."""
    articles = [
        make_article(url="https://example.com/a1", title="Article 1"),
        make_article(url="https://example.com/a2", title="Article 2"),
    ]
    store.mark_seen(articles)
    assert store.count() == 2


def test_mark_seen_is_idempotent(store: MemoryStore) -> None:
    """Marking the same article twice does not create duplicate rows."""
    article = make_article()
    store.mark_seen([article])
    store.mark_seen([article])
    assert store.count() == 1


def test_mark_seen_empty_list_is_safe(store: MemoryStore) -> None:
    """mark_seen() with an empty list does not raise."""
    store.mark_seen([])
    assert store.count() == 0


def test_mark_seen_stores_correct_fields(store: MemoryStore) -> None:
    """mark_seen() stores url, title, source_name, source_type correctly."""
    from sqlmodel import Session, select
    from tech_watch.memory.store import SeenArticle

    article = make_article(
        url="https://example.com/check-fields",
        title="Field Check Article",
        source_name="My Blog",
        source_type=SourceType.API,
    )
    store.mark_seen([article])

    with Session(store._engine) as session:
        row = session.exec(
            select(SeenArticle).where(
                SeenArticle.content_hash == article.content_hash
            )
        ).first()

    assert row is not None
    assert row.url == "https://example.com/check-fields"
    assert row.title == "Field Check Article"
    assert row.source_name == "My Blog"
    assert row.source_type == "api"
    # SQLite drops timezone info on read — seen_at is naive but correct UTC time
    assert row.seen_at is not None
    assert row.seen_at.year == 2026  # sanity check — recent datetime


# ---------------------------------------------------------------------------
# filter_unseen
# ---------------------------------------------------------------------------

def test_filter_unseen_returns_all_when_store_empty(store: MemoryStore) -> None:
    """filter_unseen() returns all articles when the store is empty."""
    articles = [
        make_article(url="https://example.com/a1"),
        make_article(url="https://example.com/a2"),
    ]
    result = store.filter_unseen(articles)
    assert len(result) == 2


def test_filter_unseen_excludes_seen_articles(store: MemoryStore) -> None:
    """filter_unseen() excludes articles already in the store."""
    seen = make_article(url="https://example.com/seen")
    new = make_article(url="https://example.com/new")

    store.mark_seen([seen])

    result = store.filter_unseen([seen, new])
    assert len(result) == 1
    assert result[0].url == "https://example.com/new"


def test_filter_unseen_returns_empty_when_all_seen(store: MemoryStore) -> None:
    """filter_unseen() returns [] when all articles have already been seen."""
    articles = [
        make_article(url="https://example.com/a1"),
        make_article(url="https://example.com/a2"),
    ]
    store.mark_seen(articles)

    result = store.filter_unseen(articles)
    assert result == []


def test_filter_unseen_preserves_order(store: MemoryStore) -> None:
    """filter_unseen() preserves the original order of unseen articles."""
    articles = [
        make_article(url=f"https://example.com/article-{i}", title=f"Article {i}")
        for i in range(5)
    ]
    # Mark articles 1 and 3 as seen
    store.mark_seen([articles[1], articles[3]])

    result = store.filter_unseen(articles)

    assert len(result) == 3
    assert result[0].url == "https://example.com/article-0"
    assert result[1].url == "https://example.com/article-2"
    assert result[2].url == "https://example.com/article-4"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def test_clear_removes_all_records(store: MemoryStore) -> None:
    """clear() removes all records from the store."""
    articles = [
        make_article(url="https://example.com/a1"),
        make_article(url="https://example.com/a2"),
    ]
    store.mark_seen(articles)
    assert store.count() == 2

    store.clear()
    assert store.count() == 0


def test_clear_allows_reprocessing(store: MemoryStore) -> None:
    """After clear(), previously seen articles can be marked seen again."""
    article = make_article()
    store.mark_seen([article])
    store.clear()

    assert store.is_seen(article.content_hash) is False
    store.mark_seen([article])
    assert store.count() == 1
