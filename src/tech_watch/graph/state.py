"""
LangGraph pipeline state.

Defines the GraphState TypedDict that flows through the pipeline nodes.
Each node receives the full state and returns a partial update.

Pipeline flow:
    collect_node → filter_node → summarize_node → digest_node → output_node

State evolution:
    collect_node    : populates raw_articles, total_collected
    filter_node     : populates scored_articles
    summarize_node  : populates summarized_articles
    digest_node     : populates digest
    output_node     : populates output_path, sets completed = True
"""

from typing import TypedDict

from tech_watch.models.article import (
    Digest,
    RawArticle,
    ScoredArticle,
    SummarizedArticle,
)


class GraphState(TypedDict, total=False):
    """
    State that flows through the LangGraph pipeline.

    All fields are optional (total=False) because each node only
    populates its own output fields — earlier fields are not yet
    available at the start of the pipeline.

    Fields are append-only across nodes — no node modifies
    what a previous node has written.
    """

    # --- collect_node output ---
    raw_articles: list[RawArticle]
    total_collected: int          # total articles fetched across all sources

    # --- filter_node output ---
    scored_articles: list[ScoredArticle]

    # --- summarize_node output ---
    summarized_articles: list[SummarizedArticle]

    # --- digest_node output ---
    digest: Digest

    # --- output_node output ---
    output_path: str              # path to the generated markdown file
    completed: bool               # True when the full pipeline succeeded

    # --- error tracking ---
    errors: list[str]             # non-fatal errors collected during the run
