"""
DigestWriter — assembles the final digest from summarized articles.

Responsibilities:
1. Groups articles by source
2. Sorts articles by relevance score within each group
3. Asks the LLM to write a global overview of the day's themes
4. Assembles a Digest object ready for markdown rendering

The global overview is the only LLM call in this agent — grouping
and sorting are deterministic operations.
"""

from collections import defaultdict

from loguru import logger
from pydantic import BaseModel, Field

from tech_watch.config.settings import Settings
from tech_watch.llm.client import LLMClient, LLMError
from tech_watch.models.article import (
    Digest,
    DigestArticleRef,
    SummarizedArticle,
    SourceType,
)


# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------

class OverviewResponse(BaseModel):
    """Expected JSON structure for the global digest overview."""
    overview: str = Field(
        description="3-4 sentence editorial overview of the day's main themes"
    )


# ---------------------------------------------------------------------------
# Source display names and icons
# ---------------------------------------------------------------------------

_SOURCE_ICONS: dict[SourceType, str] = {
    SourceType.RSS: "📰",
    SourceType.API: "🔗",
    SourceType.PDF: "🔬",
    SourceType.WEB: "🌐",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior analyst writing the introduction
to a daily curated digest. Your overview is insightful, connects themes across
articles, and highlights what matters most for the audience today.
You must respond with valid JSON only — no markdown, no explanation, raw JSON."""

_OVERVIEW_PROMPT = """Write a brief editorial overview for today's tech watch digest.

The following articles were selected as relevant today:
{article_list}

Write 3-4 sentences that:
- Identify the main themes or trends across these articles
- Highlight the most significant development of the day
- Connect ideas across different sources when relevant

Respond with a JSON object with exactly this field:
- "overview": a string containing the 3-4 sentence overview

Example:
{{"overview": "Today's digest is dominated by advances in agentic AI systems, with two arXiv papers proposing complementary approaches to multi-agent coordination. HackerNews surfaces growing practitioner interest in RAG pipeline reliability, echoing the theoretical work from academia. DevOps tooling continues to evolve around AI workload orchestration, suggesting a convergence between MLOps and traditional infrastructure practices."}}"""


# ---------------------------------------------------------------------------
# DigestWriter
# ---------------------------------------------------------------------------

class DigestWriter:
    """
    Assembles the final Digest from summarized articles.

    Groups by source, sorts by score within each group,
    generates a global LLM overview, and returns a Digest object.
    """

    def __init__(self, settings: Settings, llm_client: LLMClient) -> None:
        """
        Args:
            settings:   Application settings.
            llm_client: Configured LLM client.
        """
        self._settings = settings
        self._llm = llm_client

    async def write(
        self,
        articles: list[SummarizedArticle],
        total_collected: int = 0,
    ) -> Digest:
        """
        Assemble the final digest from summarized articles.

        Args:
            articles:        Summarized articles that passed the relevance filter.
            total_collected: Total articles collected before filtering (for stats).

        Returns:
            A Digest object ready for markdown rendering.
            Returns an empty Digest if no articles are provided.
        """
        if not articles:
            logger.warning("DigestWriter: no articles to write digest from")
            return Digest(total_collected=total_collected, total_filtered=0)

        logger.info(f"DigestWriter: assembling digest from {len(articles)} articles")

        # Step 1 — convert to DigestArticleRef and group by source
        refs = [self._to_ref(article) for article in articles]
        grouped = self._group_by_source(refs)

        # Step 2 — sort within each group by relevance score descending
        sorted_refs: list[DigestArticleRef] = []
        for source_name in sorted(grouped.keys(), key=str.lower):
            group = sorted(
                grouped[source_name],
                key=lambda a: a.relevance_score,
                reverse=True,
            )
            sorted_refs.extend(group)

        # Step 3 — generate global overview via LLM
        overview = await self._generate_overview(sorted_refs)

        digest = Digest(
            articles=sorted_refs,
            global_summary=overview,
            total_collected=total_collected,
            total_filtered=len(articles),
        )

        logger.info(
            f"DigestWriter: digest ready — "
            f"{len(sorted_refs)} articles from {len(grouped)} source(s)"
        )
        return digest

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _to_ref(self, article: SummarizedArticle) -> DigestArticleRef:
        """Convert a SummarizedArticle to a lightweight DigestArticleRef."""
        return DigestArticleRef(
            url=article.url,
            title=article.title,
            source_name=article.source_name,
            source_type=article.source_type,
            relevance_score=article.relevance_score,
            summary=article.summary,
            key_points=article.key_points,
            matched_topics=article.matched_topics,
            published_at=article.published_at,
        )

    def _group_by_source(
        self, refs: list[DigestArticleRef]
    ) -> dict[str, list[DigestArticleRef]]:
        """Group DigestArticleRef instances by source_name."""
        groups: dict[str, list[DigestArticleRef]] = defaultdict(list)
        for ref in refs:
            groups[ref.source_name].append(ref)
        return dict(groups)

    async def _generate_overview(
        self, refs: list[DigestArticleRef]
    ) -> str:
        """
        Ask the LLM to write a global editorial overview of the digest.

        Falls back to an empty string if the LLM call fails — the digest
        remains usable without the overview.
        """
        # Build a compact article list for the prompt
        article_list = "\n".join(
            f"- [{ref.source_name}] {ref.title} "
            f"(topics: {', '.join(ref.matched_topics) or 'general'})"
            for ref in refs
        )

        prompt = _OVERVIEW_PROMPT.format(article_list=article_list)

        try:
            response = await self._llm.complete_json(
                prompt=prompt,
                schema=OverviewResponse,
                system_prompt=_SYSTEM_PROMPT,
            )
            return response.overview
        except LLMError as e:
            logger.warning(f"DigestWriter: failed to generate overview: {e}")
            return ""
