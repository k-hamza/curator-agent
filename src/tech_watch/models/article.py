"""
Data contracts between pipeline agents.

Flow:
    Collectors   → RawArticle
    FilterAgent  → ScoredArticle
    SummaryAgent → SummarizedArticle
    DigestWriter → Digest
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    """
    Identifies the collection mechanism used to produce a RawArticle.
    Tied to HOW the article was fetched, not to the specific source.

    RSS — anything parsed via feedparser (blogs, news feeds...)
    API — sources with a structured API (HackerNews, Reddit...)
    PDF — documents downloaded and extracted from PDF (arXiv papers...)
    WEB — generic HTML scraping
    """
    RSS = "rss"
    API = "api"
    PDF = "pdf"
    WEB = "web"


# ---------------------------------------------------------------------------
# Stage 1 — Collector output
# ---------------------------------------------------------------------------

class RawArticle(BaseModel):
    """
    Raw article as collected from a source, before any LLM processing.
    Immutable once created by a collector.
    """

    # --- Identity ---
    url: str = Field(description="Canonical URL of the article")
    title: str = Field(description="Article title")
    source_type: SourceType = Field(description="Collection mechanism used")
    source_name: str = Field(description="Human-readable source name, e.g. 'Hugging Face Blog'")

    # --- Content ---
    content: str = Field(description="Full text content or abstract (plain text, no HTML)")
    authors: list[str] = Field(default_factory=list, description="Author names, if available")

    # --- Metadata ---
    published_at: datetime | None = Field(
        default=None,
        description="Original publication date from the source (UTC)"
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this article was collected (UTC)"
    )

    # --- Generic attachment (PDF or any downloaded file) ---
    attachment_path: str | None = Field(
        default=None,
        description="Local path to a downloaded file (e.g. PDF for arXiv papers)"
    )

    # --- Generic external identifier ---
    external_id: str | None = Field(
        default=None,
        description=(
            "Source-specific identifier when useful "
            "(e.g. '2401.12345' for arXiv, post ID for HackerNews)"
        )
    )

    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the URL — used for deduplication in memory store."""
        import hashlib
        return hashlib.sha256(self.url.encode()).hexdigest()

    model_config = {"frozen": True}  # immutable after creation


# ---------------------------------------------------------------------------
# Stage 2 — FilterAgent output
# ---------------------------------------------------------------------------

class ScoredArticle(RawArticle):
    """
    RawArticle enriched with a relevance score and matched topics.
    Produced by FilterAgent after LLM evaluation.
    """

    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="LLM-assigned relevance score between 0.0 and 1.0"
    )
    matched_topics: list[str] = Field(
        default_factory=list,
        description="Interest topics matched by the LLM, e.g. ['LLM', 'agents']"
    )
    filter_reasoning: str = Field(
        default="",
        description="LLM explanation for the assigned score (useful for debugging)"
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Stage 3 — SummaryAgent output
# ---------------------------------------------------------------------------

class SummarizedArticle(ScoredArticle):
    """
    ScoredArticle enriched with a human-readable summary.
    Produced by SummaryAgent after LLM summarization.
    """

    summary: str = Field(description="LLM-generated summary (2-4 sentences)")
    key_points: list[str] = Field(
        default_factory=list,
        description="3-5 key takeaways extracted by the LLM"
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Stage 4 — DigestWriter output
# ---------------------------------------------------------------------------

class DigestArticleRef(BaseModel):
    """
    Lightweight reference to an article inside a Digest.
    Avoids duplicating full content in the Digest object.
    """
    url: str
    title: str
    source_name: str
    source_type: SourceType
    relevance_score: float
    summary: str
    key_points: list[str]
    matched_topics: list[str]
    published_at: datetime | None = None

    model_config = {"frozen": True}


class Digest(BaseModel):
    """
    Final output of the pipeline — one digest per run.
    Contains all selected articles and a global LLM-written summary.
    """

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this digest was generated (UTC)"
    )
    articles: list[DigestArticleRef] = Field(
        default_factory=list,
        description="Selected and summarized articles, ordered by relevance score"
    )
    global_summary: str = Field(
        default="",
        description="LLM-written overview of today's main themes across all articles"
    )
    total_collected: int = Field(
        default=0,
        description="Total articles collected before filtering (for stats)"
    )
    total_filtered: int = Field(
        default=0,
        description="Articles that passed the relevance threshold"
    )

    @computed_field
    @property
    def sources_used(self) -> list[str]:
        """Deduplicated list of source names present in this digest."""
        return sorted({a.source_name for a in self.articles})

    model_config = {"frozen": True}
