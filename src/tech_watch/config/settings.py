"""
Configuration loading and validation.

Reads config.yaml from the project root, validates every value with Pydantic,
and exposes a single Settings object used across the entire application.

Usage:
    from tech_watch.config.settings import load_settings

    settings = load_settings()               # loads config.yaml at project root
    settings = load_settings("custom.yaml")  # loads a custom config file
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from tech_watch.models.article import SourceType


# ---------------------------------------------------------------------------
# Agent settings
# ---------------------------------------------------------------------------

class AgentSettings(BaseModel):
    """LLM and pipeline behaviour settings."""

    model: str = Field(description="Ollama model name, e.g. 'qwen3:8b'")
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL of the Ollama OpenAI-compatible API"
    )
    max_articles_per_run: int = Field(
        default=50, ge=1,
        description="Maximum total articles to process per run (across all sources)"
    )
    relevance_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum relevance score to include an article in the digest"
    )

    # --- PDF settings ---
    pdf_timeout: float = Field(
        default=30.0, gt=0,
        description="HTTP timeout in seconds for PDF downloads"
    )
    pdf_max_chars: int = Field(
        default=8000, ge=500,
        description=(
            "Maximum characters extracted per PDF — "
            "reduce if your model has a small context window"
        )
    )
    pdf_max_concurrent: int = Field(
        default=3, ge=1, le=10,
        description="Maximum number of PDFs downloaded in parallel"
    )
    pdf_storage_dir: str = Field(
        default="data/pdfs",
        description=(
            "Base directory for downloaded PDFs. "
            "Each source gets its own subdirectory: {pdf_storage_dir}/{source_name}/"
        )
    )


# ---------------------------------------------------------------------------
# Scheduling settings
# ---------------------------------------------------------------------------

class SchedulingSettings(BaseModel):
    """Scheduling configuration."""

    enabled: bool = Field(default=False)
    cron: str = Field(
        default="0 7 * * *",
        description="Cron expression for the daily digest trigger"
    )

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        """Ensure the cron expression has exactly 5 fields."""
        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression '{v}': expected 5 fields "
                f"(minute hour day month weekday), got {len(parts)}"
            )
        return v


# ---------------------------------------------------------------------------
# Source settings — one model per collection mechanism
# ---------------------------------------------------------------------------

class BaseSourceSettings(BaseModel):
    """Common fields for every source, regardless of type."""

    name: str = Field(description="Human-readable source name")
    type: SourceType = Field(description="Collection mechanism (rss, api, pdf, web)")
    url: str = Field(description="Base URL of the source")
    enabled: bool = Field(default=True)
    max_items: int = Field(
        default=20, ge=1,
        description="Maximum items to fetch per run for this source"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Optional category filter — meaning is source-specific"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional source-specific parameters not covered above"
    )

    model_config = {"extra": "allow"}


class RssSourceSettings(BaseSourceSettings):
    """RSS/Atom feed source — type must be SourceType.RSS."""

    @model_validator(mode="after")
    def check_type(self) -> "RssSourceSettings":
        if self.type != SourceType.RSS:
            raise ValueError(f"RssSourceSettings requires type='rss', got '{self.type}'")
        return self


class ApiSourceSettings(BaseSourceSettings):
    """API-based source (HackerNews, etc.) — type must be SourceType.API."""

    @model_validator(mode="after")
    def check_type(self) -> "ApiSourceSettings":
        if self.type != SourceType.API:
            raise ValueError(f"ApiSourceSettings requires type='api', got '{self.type}'")
        return self


class PdfSourceSettings(BaseSourceSettings):
    """PDF document source (arXiv, etc.) — type must be SourceType.PDF."""

    @model_validator(mode="after")
    def check_type(self) -> "PdfSourceSettings":
        if self.type != SourceType.PDF:
            raise ValueError(f"PdfSourceSettings requires type='pdf', got '{self.type}'")
        return self


class WebSourceSettings(BaseSourceSettings):
    """Generic HTML scraping source — type must be SourceType.WEB."""

    @model_validator(mode="after")
    def check_type(self) -> "WebSourceSettings":
        if self.type != SourceType.WEB:
            raise ValueError(f"WebSourceSettings requires type='web', got '{self.type}'")
        return self


# Map SourceType → concrete settings class
_SOURCE_TYPE_MAP: dict[SourceType, type[BaseSourceSettings]] = {
    SourceType.RSS: RssSourceSettings,
    SourceType.API: ApiSourceSettings,
    SourceType.PDF: PdfSourceSettings,
    SourceType.WEB: WebSourceSettings,
}


def _parse_source(data: dict[str, Any]) -> BaseSourceSettings:
    """
    Dispatch a raw source dict to the correct settings class based on 'type'.
    Called during Settings validation for each entry in the sources list.
    """
    raw_type = data.get("type")
    if not raw_type:
        raise ValueError(f"Source '{data.get('name', '?')}' is missing required field 'type'")

    try:
        source_type = SourceType(raw_type)
    except ValueError:
        valid = [t.value for t in SourceType]
        raise ValueError(
            f"Source '{data.get('name', '?')}' has unknown type '{raw_type}'. "
            f"Valid types: {valid}"
        )

    cls = _SOURCE_TYPE_MAP[source_type]
    return cls(**data)


# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    """
    Root configuration object.
    Built from config.yaml via load_settings().
    """

    agent: AgentSettings
    interests: list[str] = Field(
        min_length=1,
        description="Topics used by the FilterAgent to score article relevance"
    )
    scheduling: SchedulingSettings = Field(default_factory=SchedulingSettings)
    sources: list[BaseSourceSettings] = Field(
        default_factory=list,
        description="All configured sources — one entry per source"
    )

    @field_validator("interests")
    @classmethod
    def interests_not_empty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [i.strip() for i in v if i.strip()]
        if not cleaned:
            raise ValueError("interests list must contain at least one non-empty string")
        return cleaned

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v: list[dict[str, Any]]) -> list[BaseSourceSettings]:
        """Dispatch each source dict to its concrete settings class."""
        if not isinstance(v, list):
            raise ValueError("'sources' must be a list")
        return [_parse_source(entry) for entry in v]

    @model_validator(mode="after")
    def at_least_one_source_enabled(self) -> "Settings":
        if not any(s.enabled for s in self.sources):
            raise ValueError(
                "At least one source must be enabled in the sources list."
            )
        return self

    def sources_by_type(self, source_type: SourceType) -> list[BaseSourceSettings]:
        """Return all enabled sources of a given type."""
        return [s for s in self.sources if s.type == source_type and s.enabled]


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

# Default config path: project root / config.yaml
# __file__ is src/tech_watch/config/settings.py → .parent x4 = project root
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config.yaml"


def load_settings(config_path: Path | str | None = None) -> Settings:
    """
    Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to config.yaml at the project root.

    Returns:
        A validated Settings instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is invalid or missing required fields.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} is empty or not a valid YAML mapping")

    return Settings(**raw)
