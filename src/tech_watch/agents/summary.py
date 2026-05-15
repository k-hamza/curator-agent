"""
SummaryAgent — LLM-based article summarization.

Receives ScoredArticle instances that passed the relevance threshold
and produces SummarizedArticle instances with:
- A 2-4 sentence summary in the article's original language
- 3-5 key takeaways

Only called on articles that passed FilterAgent — no wasted LLM calls
on irrelevant content.
"""

from loguru import logger
from pydantic import BaseModel, Field

from tech_watch.config.settings import Settings
from tech_watch.llm.client import LLMClient, LLMError
from tech_watch.models.article import ScoredArticle, SummarizedArticle


# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------

class SummaryResponse(BaseModel):
    """Expected JSON structure from the LLM for each article summarized."""
    summary: str = Field(description="2-4 sentence summary of the article")
    key_points: list[str] = Field(
        description="3-5 key takeaways from the article"
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert analyst writing summaries for a daily curated digest.
Your summaries are concise, accurate, and highlight what matters most for a technical audience.
You must respond with valid JSON only — no markdown, no explanation, raw JSON."""

_SUMMARY_PROMPT = """Summarise the following article for a tech watch digest.

Article title: {title}
Article source: {source_name}
Article content:
{content}

Write the summary in the SAME LANGUAGE as the article content.

Respond with a JSON object with exactly these fields:
- "summary": a 2-4 sentence summary capturing the main contribution or news
- "key_points": a list of 3-5 short bullet points (strings) highlighting key takeaways

Example response:
{{
  "summary": "Researchers propose a new framework for multi-agent LLM systems that reduces coordination overhead by 40%. The approach uses a hierarchical planning mechanism where a supervisor agent delegates subtasks to specialised sub-agents.",
  "key_points": [
    "Hierarchical planning reduces inter-agent communication by 40%",
    "Supervisor-agent architecture improves task decomposition",
    "Evaluated on 3 benchmarks, outperforming flat agent architectures",
    "Open-source implementation available on GitHub"
  ]
}}"""


# ---------------------------------------------------------------------------
# SummaryAgent
# ---------------------------------------------------------------------------

class SummaryAgent:
    """
    Summarizes articles that passed the relevance filter.

    Produces a concise summary and key points for each article,
    in the article's original language.
    """

    def __init__(self, settings: Settings, llm_client: LLMClient) -> None:
        """
        Args:
            settings:   Application settings (unused currently, kept for consistency).
            llm_client: Configured LLM client.
        """
        self._settings = settings
        self._llm = llm_client

    async def summarize(
        self, articles: list[ScoredArticle]
    ) -> list[SummarizedArticle]:
        """
        Summarize all scored articles.

        Articles where the LLM call fails are skipped — a failed summary
        is not worth including in the digest without content.

        Args:
            articles: Scored articles that passed the relevance threshold.

        Returns:
            List of SummarizedArticle instances, in the same order as input.
        """
        if not articles:
            return []

        logger.info(f"SummaryAgent: summarizing {len(articles)} articles")

        summarized: list[SummarizedArticle] = []
        failed = 0

        for article in articles:
            result = await self._summarize_article(article)
            if result is not None:
                summarized.append(result)
            else:
                failed += 1

        logger.info(
            f"SummaryAgent: {len(summarized)} summarized, {failed} failed"
        )
        return summarized

    async def _summarize_article(
        self, article: ScoredArticle
    ) -> SummarizedArticle | None:
        """
        Summarize a single article using the LLM.

        Returns None if the LLM call fails — the article is dropped
        rather than included with empty content.
        """
        prompt = _SUMMARY_PROMPT.format(
            title=article.title,
            source_name=article.source_name,
            content=article.content,
        )

        try:
            response = await self._llm.complete_json(
                prompt=prompt,
                schema=SummaryResponse,
                system_prompt=_SYSTEM_PROMPT,
            )
        except LLMError as e:
            logger.warning(
                f"SummaryAgent: LLM error summarizing "
                f"'{article.title[:60]}': {e}"
            )
            return None

        return SummarizedArticle(
            **article.model_dump(),
            summary=response.summary,
            key_points=response.key_points,
        )
