"""
FilterAgent — LLM-based article relevance scoring.

Scores every collected article against the user's interests.
Articles scoring below the relevance_threshold are discarded.

The agent uses a structured JSON response from the LLM:
    {
        "score": 0.85,
        "matched_topics": ["LLM", "agents"],
        "reasoning": "The article directly addresses LLM agent architectures"
    }

Only articles above settings.agent.relevance_threshold are returned.
"""

from pydantic import BaseModel, Field
from loguru import logger

from tech_watch.config.settings import Settings
from tech_watch.llm.client import LLMClient, LLMError
from tech_watch.models.article import RawArticle, ScoredArticle


# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------

class FilterResponse(BaseModel):
    """Expected JSON structure from the LLM for each article scored."""
    score: float = Field(ge=0.0, le=1.0)
    matched_topics: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert content analyst for a content curation agent.
Your task is to evaluate whether an article is relevant to a user's interests.
You must respond with valid JSON only — no markdown, no explanation, raw JSON."""

_FILTER_PROMPT = """Evaluate the relevance of the following article to the user's interests.

User interests: {interests}

IMPORTANT RULES:
- An article is relevant if it matches AT LEAST ONE interest — it does NOT need to cover all of them.
- Score based on how well the article covers its matched interest(s), not on how many interests it covers.
- A focused article on a single interest should score as high as an article covering multiple interests.

Article title: {title}
Article content (excerpt):
{content}

Respond with a JSON object with exactly these fields:
- "score": a float between 0.0 (completely irrelevant) and 1.0 (perfectly relevant)
- "matched_topics": list of strings — which interests from the list are matched (can be just one)
- "reasoning": one sentence explaining the score

Examples:
{{"score": 0.92, "matched_topics": ["robotique"], "reasoning": "The article focuses entirely on robotics control systems, directly matching the robotique interest."}}
{{"score": 0.88, "matched_topics": ["LLM", "agentique"], "reasoning": "Covers both LLM fine-tuning and agent orchestration patterns."}}
{{"score": 0.03, "matched_topics": [], "reasoning": "Article about cooking recipes, no relation to any configured interest."}}"""

# Maximum content characters sent to the LLM for filtering
# Shorter than pdf_max_chars — we only need enough to judge relevance
_FILTER_CONTENT_MAX_CHARS = 1500


# ---------------------------------------------------------------------------
# FilterAgent
# ---------------------------------------------------------------------------

class FilterAgent:
    """
    Scores articles for relevance using an LLM.

    Articles below the relevance_threshold are discarded.
    Articles above are returned as ScoredArticle instances.
    """

    def __init__(self, settings: Settings, llm_client: LLMClient) -> None:
        """
        Args:
            settings:   Application settings — provides interests and threshold.
            llm_client: Configured LLM client.
        """
        self._settings = settings
        self._llm = llm_client
        self._threshold = settings.agent.relevance_threshold
        self._interests = settings.interests

    async def filter(self, articles: list[RawArticle]) -> list[ScoredArticle]:
        """
        Score all articles and return those above the relevance threshold.

        Args:
            articles: Raw articles from collectors, after deduplication.

        Returns:
            List of ScoredArticle instances, filtered and sorted by score
            descending within each source group (sorting done in DigestWriter).
        """
        if not articles:
            return []

        logger.info(f"FilterAgent: scoring {len(articles)} articles")

        scored: list[ScoredArticle] = []
        discarded = 0

        for article in articles:
            result = await self._score_article(article)
            if result is None:
                discarded += 1
                continue

            if result.relevance_score >= self._threshold:
                scored.append(result)
            else:
                logger.debug(
                    f"Discarded (score={result.relevance_score:.2f}): {article.source_name} : {article.title[:60]}"
                )
                discarded += 1

        logger.info(
            f"FilterAgent: {len(scored)} relevant, {discarded} discarded "
            f"(threshold={self._threshold})"
        )
        return scored

    async def _score_article(self, article: RawArticle) -> ScoredArticle | None:
        """
        Score a single article using the LLM.

        Returns None if the LLM call fails — the article is silently discarded
        rather than blocking the pipeline.
        """
        # Truncate content for the filter prompt — we don't need the full text
        content_excerpt = article.content[:_FILTER_CONTENT_MAX_CHARS]
        if len(article.content) > _FILTER_CONTENT_MAX_CHARS:
            content_excerpt += "..."

        prompt = _FILTER_PROMPT.format(
            interests=", ".join(self._interests),
            title=article.title,
            content=content_excerpt,
        )

        try:
            response = await self._llm.complete_json(
                prompt=prompt,
                schema=FilterResponse,
                system_prompt=_SYSTEM_PROMPT,
            )
        except LLMError as e:
            logger.warning(
                f"FilterAgent: LLM error scoring '{article.title[:60]}': {e}"
            )
            return None

        return ScoredArticle(
            **article.model_dump(),
            relevance_score=response.score,
            matched_topics=response.matched_topics,
            filter_reasoning=response.reasoning,
        )
