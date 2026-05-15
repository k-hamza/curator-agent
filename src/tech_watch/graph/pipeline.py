"""
LangGraph pipeline — assembles all nodes into a deterministic StateGraph.

Pipeline flow:
    collect → filter → summarize → digest → output → END

Each node is a pure async function that receives the full GraphState
and returns a partial state update (only the fields it populates).

Usage:
    from tech_watch.graph.pipeline import build_pipeline
    from tech_watch.config.settings import load_settings

    settings = load_settings()
    pipeline = build_pipeline(settings)
    final_state = await pipeline.ainvoke({})
"""

from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from loguru import logger

from tech_watch.agents.digest import DigestWriter
from tech_watch.agents.filter import FilterAgent
from tech_watch.agents.summary import SummaryAgent
from tech_watch.collectors.base import autodiscover_collectors, registry
from tech_watch.config.settings import Settings
from tech_watch.graph.state import GraphState
from tech_watch.llm.client import LLMClient
from tech_watch.memory.store import MemoryStore
from tech_watch.models.article import RawArticle
from tech_watch.output.markdown import MarkdownWriter


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _make_collect_node(settings: Settings, store: MemoryStore):
    """
    Build the collect node function.

    Runs all enabled collectors concurrently, deduplicates results
    against the memory store, and returns raw unseen articles.
    """
    import asyncio

    async def collect_node(state: GraphState) -> dict:
        logger.info("=== Pipeline: COLLECT ===")

        autodiscover_collectors()
        collector_pairs = registry.get_enabled(settings)

        if not collector_pairs:
            logger.warning("No collectors available — check your config.yaml")
            return {
                "raw_articles": [],
                "total_collected": 0,
                "errors": state.get("errors", []) + ["No collectors available"],
            }

        # Run all collectors concurrently
        tasks = [
            collector.safe_collect(source)
            for source, collector in collector_pairs
        ]
        results = await asyncio.gather(*tasks)

        # Flatten results from all sources
        all_articles: list[RawArticle] = []
        for articles in results:
            all_articles.extend(articles)

        total_collected = len(all_articles)
        logger.info(f"Collected {total_collected} articles across all sources")

        # Respect max_articles_per_run
        if total_collected > settings.agent.max_articles_per_run:
            logger.info(
                f"Capping at {settings.agent.max_articles_per_run} articles "
                f"(collected {total_collected})"
            )
            all_articles = all_articles[: settings.agent.max_articles_per_run]

        # Deduplicate against memory store
        unseen = store.filter_unseen(all_articles)
        logger.info(
            f"After deduplication: {len(unseen)} new articles "
            f"({total_collected - len(unseen)} already seen)"
        )

        return {
            "raw_articles": unseen,
            "total_collected": total_collected,
            "errors": state.get("errors", []),
        }

    return collect_node


def _make_filter_node(settings: Settings, llm_client: LLMClient):
    """Build the filter node function."""

    async def filter_node(state: GraphState) -> dict:
        logger.info("=== Pipeline: FILTER ===")

        raw_articles = state.get("raw_articles", [])
        if not raw_articles:
            logger.info("No articles to filter")
            return {"scored_articles": []}

        agent = FilterAgent(settings=settings, llm_client=llm_client)
        scored = await agent.filter(raw_articles)

        return {"scored_articles": scored}

    return filter_node


def _make_summarize_node(settings: Settings, llm_client: LLMClient):
    """Build the summarize node function."""

    async def summarize_node(state: GraphState) -> dict:
        logger.info("=== Pipeline: SUMMARIZE ===")

        scored_articles = state.get("scored_articles", [])
        if not scored_articles:
            logger.info("No articles to summarize")
            return {"summarized_articles": []}

        agent = SummaryAgent(settings=settings, llm_client=llm_client)
        summarized = await agent.summarize(scored_articles)

        return {"summarized_articles": summarized}

    return summarize_node


def _make_digest_node(settings: Settings, llm_client: LLMClient):
    """Build the digest node function."""

    async def digest_node(state: GraphState) -> dict:
        logger.info("=== Pipeline: DIGEST ===")

        summarized_articles = state.get("summarized_articles", [])
        total_collected = state.get("total_collected", 0)

        writer = DigestWriter(settings=settings, llm_client=llm_client)
        digest = await writer.write(
            articles=summarized_articles,
            total_collected=total_collected,
        )

        return {"digest": digest}

    return digest_node


def _make_output_node(settings: Settings, store: MemoryStore):
    """
    Build the output node function.

    Writes the digest to a markdown file and marks articles as seen
    in the memory store.
    """

    async def output_node(state: GraphState) -> dict:
        logger.info("=== Pipeline: OUTPUT ===")

        digest = state.get("digest")
        summarized_articles = state.get("summarized_articles", [])

        if not digest or not digest.articles:
            logger.warning("No digest to write")
            return {"completed": False, "output_path": ""}

        # Write markdown file
        output_dir = Path("digests")
        md_writer = MarkdownWriter(output_dir=output_dir)
        output_path = md_writer.write(digest)
        logger.info(f"Digest written to: {output_path}")

        # Mark processed articles as seen in memory store
        if summarized_articles:
            store.mark_seen(summarized_articles)
            logger.info(f"Marked {len(summarized_articles)} articles as seen")

        return {
            "output_path": str(output_path),
            "completed": True,
        }

    return output_node


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    settings: Settings,
    store: MemoryStore | None = None,
    llm_client: LLMClient | None = None,
) -> StateGraph:
    """
    Build and compile the LangGraph pipeline.

    Args:
        settings:   Validated application settings.
        store:      Memory store for deduplication. Created from settings if None.
        llm_client: LLM client. Created from settings if None.

    Returns:
        Compiled LangGraph StateGraph ready to invoke with ainvoke({}).
    """
    # Resolve dependencies
    if store is None:
        store = MemoryStore(db_path=settings.agent.__dict__.get(
            "memory_db_path", "data/memory.db"
        ))
        store.init()

    if llm_client is None:
        llm_client = LLMClient.from_settings(settings)

    # Build the graph
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("collect", _make_collect_node(settings, store))
    graph.add_node("filter", _make_filter_node(settings, llm_client))
    graph.add_node("summarize", _make_summarize_node(settings, llm_client))
    graph.add_node("digest", _make_digest_node(settings, llm_client))
    graph.add_node("output", _make_output_node(settings, store))

    # Define edges — deterministic linear flow
    graph.set_entry_point("collect")
    graph.add_edge("collect", "filter")
    graph.add_edge("filter", "summarize")
    graph.add_edge("summarize", "digest")
    graph.add_edge("digest", "output")
    graph.add_edge("output", END)

    return graph.compile()
