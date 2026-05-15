"""
Persistent memory store for article deduplication.

Uses SQLite via SQLModel to track which articles have already been
processed. Before any LLM call, the pipeline checks this store to
avoid reprocessing the same article across runs.

Schema:
    SeenArticle — one row per processed article
        content_hash : SHA-256 of the article URL (primary key)
        url          : original article URL
        title        : article title
        source_name  : human-readable source name
        source_type  : SourceType value (rss, api, pdf, web)
        seen_at      : UTC datetime when first processed

Usage:
    from tech_watch.memory.store import MemoryStore

    store = MemoryStore("data/memory.db")
    store.init()

    new_articles = store.filter_unseen(articles)
    store.mark_seen(new_articles)
"""

from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from sqlmodel import Field, Session, SQLModel, create_engine, select


# ---------------------------------------------------------------------------
# Database model
# ---------------------------------------------------------------------------

class SeenArticle(SQLModel, table=True):
    """
    Represents an article that has already been processed by the pipeline.
    One row per article, keyed by content_hash (SHA-256 of the URL).
    """

    content_hash: str = Field(primary_key=True)
    url: str
    title: str
    source_name: str
    source_type: str  # stored as string — SourceType.value
    seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        # Note: SQLite strips timezone info on read — seen_at is stored as
        # naive UTC datetime. Always treat it as UTC when reading back.
    )


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    SQLite-backed store for tracking processed articles.

    Thread-safe for single-process use (SQLite WAL mode enabled).
    Not designed for concurrent multi-process access.
    """

    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        """
        Args:
            db_path: Path to the SQLite database file.
                     Parent directory is created automatically if needed.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        db_url = f"sqlite:///{self._db_path}"
        # WAL mode: allows concurrent reads while writing
        self._engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )

    def init(self) -> None:
        """
        Create database tables if they don't exist.
        Safe to call multiple times (idempotent).
        Called once at application startup.
        """
        SQLModel.metadata.create_all(self._engine)
        logger.debug(f"Memory store initialised at {self._db_path}")

    # ---------------------------------------------------------------------------
    # Core operations
    # ---------------------------------------------------------------------------

    def is_seen(self, content_hash: str) -> bool:
        """
        Check if an article has already been processed.

        Args:
            content_hash: SHA-256 hash of the article URL.

        Returns:
            True if the article is in the store, False otherwise.
        """
        with Session(self._engine) as session:
            result = session.get(SeenArticle, content_hash)
            return result is not None

    def filter_unseen(self, articles: list) -> list:
        """
        Return only articles that have NOT been processed before.

        Args:
            articles: List of RawArticle (or any model with content_hash).

        Returns:
            Filtered list — articles not present in the store.
        """
        unseen = [a for a in articles if not self.is_seen(a.content_hash)]

        seen_count = len(articles) - len(unseen)
        if seen_count > 0:
            logger.info(
                f"Deduplication: {seen_count} already seen, "
                f"{len(unseen)} new articles"
            )

        return unseen

    def mark_seen(self, articles: list) -> None:
        """
        Record articles as processed in the store.

        Silently skips articles already present (no duplicate key error).

        Args:
            articles: List of RawArticle (or any model with the required fields).
        """
        if not articles:
            return

        with Session(self._engine) as session:
            added = 0
            for article in articles:
                # Skip if already present — can happen if pipeline reruns
                existing = session.get(SeenArticle, article.content_hash)
                if existing is not None:
                    continue

                seen = SeenArticle(
                    content_hash=article.content_hash,
                    url=article.url,
                    title=article.title,
                    source_name=article.source_name,
                    source_type=article.source_type.value,
                )
                session.add(seen)
                added += 1

            session.commit()

        logger.debug(f"Marked {added} article(s) as seen")

    # ---------------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of articles in the store."""
        with Session(self._engine) as session:
            return len(session.exec(select(SeenArticle)).all())

    def clear(self) -> None:
        """
        Delete all records from the store.
        Intended for testing and maintenance only.
        """
        with Session(self._engine) as session:
            seen_articles = session.exec(select(SeenArticle)).all()
            for article in seen_articles:
                session.delete(article)
            session.commit()
        logger.warning("Memory store cleared")
