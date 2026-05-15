"""
Markdown digest writer.

Renders a Digest object into a formatted markdown file
saved in the digests/ directory.

Output filename format: YYYY-MM-DD.md
If a digest already exists for today, it is overwritten.

Markdown structure:
    # Tech Watch — DD Month YYYY
    > Stats line

    ## Overview
    [global summary]

    ## Articles

    ### {icon} {source_name}
    #### [{title}]({url})
    ...

Usage:
    from tech_watch.output.markdown import MarkdownWriter
    from pathlib import Path

    writer = MarkdownWriter(output_dir=Path("digests"))
    output_path = writer.write(digest)
"""

from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from tech_watch.agents.digest import _SOURCE_ICONS
from tech_watch.models.article import Digest, DigestArticleRef, SourceType


class MarkdownWriter:
    """
    Renders a Digest object into a markdown file.
    """

    def __init__(self, output_dir: Path = Path("digests")) -> None:
        """
        Args:
            output_dir: Directory where digest files are written.
                        Created automatically if it doesn't exist.
        """
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, digest: Digest) -> Path:
        """
        Render the digest to a markdown file.

        Args:
            digest: Fully assembled Digest object.

        Returns:
            Path to the written markdown file.
        """
        content = self._render(digest)
        output_path = self._output_path(digest.generated_at)
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"Digest written: {output_path} ({len(content)} chars)")
        return output_path

    # ---------------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------------

    def _render(self, digest: Digest) -> str:
        """Render the full digest as a markdown string."""
        sections: list[str] = []

        # Header
        sections.append(self._render_header(digest))

        # Overview
        if digest.global_summary:
            sections.append(self._render_overview(digest.global_summary))

        # Articles grouped by source
        sections.append(self._render_articles(digest))

        # Stats footer
        sections.append(self._render_stats(digest))

        return "\n\n".join(sections) + "\n"

    def _render_header(self, digest: Digest) -> str:
        """Render the digest title and stats line."""
        date_str = digest.generated_at.strftime("%d %B %Y")
        sources = ", ".join(digest.sources_used) if digest.sources_used else "none"

        return (
            f"# Tech Watch — {date_str}\n\n"
            f"> **{digest.total_filtered}** articles selected "
            f"from **{digest.total_collected}** collected · "
            f"Sources: {sources}"
        )

    def _render_overview(self, global_summary: str) -> str:
        """Render the global LLM overview section."""
        return f"## Overview\n\n{global_summary}"

    def _render_articles(self, digest: Digest) -> str:
        """Render all articles grouped by source."""
        if not digest.articles:
            return "## Articles\n\n_No articles selected for today._"

        # Group by source_name preserving the order from the digest
        seen_sources: list[str] = []
        groups: dict[str, list[DigestArticleRef]] = {}
        for article in digest.articles:
            if article.source_name not in groups:
                groups[article.source_name] = []
                seen_sources.append(article.source_name)
            groups[article.source_name].append(article)

        lines: list[str] = ["## Articles"]

        for source_name in seen_sources:
            icon = _SOURCE_ICONS.get(
                groups[source_name][0].source_type, "📄"
            )
            lines.append(f"\n### {icon} {source_name}")

            for article in groups[source_name]:
                lines.append(self._render_article(article))

        return "\n".join(lines)

    def _render_article(self, article: DigestArticleRef) -> str:
        """Render a single article entry."""
        lines: list[str] = []

        # Title as link
        lines.append(f"\n#### [{article.title}]({article.url})")

        # Metadata line
        meta_parts: list[str] = []
        meta_parts.append(f"**Score** {article.relevance_score:.0%}")
        if article.matched_topics:
            meta_parts.append(f"**Topics** {', '.join(article.matched_topics)}")
        if article.published_at:
            date_str = article.published_at.strftime("%d %b %Y")
            meta_parts.append(f"**Published** {date_str}")
        if article.authors if hasattr(article, 'authors') else False:
            pass  # authors not in DigestArticleRef — omitted by design

        lines.append(" · ".join(meta_parts))

        # Summary
        lines.append(f"\n{article.summary}")

        # Key points
        if article.key_points:
            lines.append("")
            for point in article.key_points:
                lines.append(f"- {point}")

        lines.append("\n---")

        return "\n".join(lines)

    def _render_stats(self, digest: Digest) -> str:
        """Render the footer stats section."""
        generated_str = digest.generated_at.strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"## Stats\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Articles collected | {digest.total_collected} |\n"
            f"| Articles selected | {digest.total_filtered} |\n"
            f"| Sources | {len(digest.sources_used)} |\n"
            f"| Generated at | {generated_str} |"
        )

    # ---------------------------------------------------------------------------
    # File path
    # ---------------------------------------------------------------------------

    def _output_path(self, generated_at: datetime) -> Path:
        """
        Return the output file path for a given generation datetime.
        Format: digests/YYYY-MM-DD.md
        """
        date_str = generated_at.strftime("%Y-%m-%d_%H-%M")
        return self._output_dir / f"{date_str}.md"
