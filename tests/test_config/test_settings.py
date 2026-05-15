"""
Tests for configuration loading and validation.

Uses temporary YAML files to avoid depending on the real config.yaml,
so tests remain isolated and reproducible.
"""

import pytest
from pathlib import Path

from tech_watch.config.settings import load_settings
from tech_watch.models.article import SourceType


# ---------------------------------------------------------------------------
# Fixtures — reusable YAML content
# ---------------------------------------------------------------------------

VALID_CONFIG = """
agent:
  model: "qwen3:8b"
  llm_base_url: "http://localhost:11434/v1"
  max_articles_per_run: 50
  relevance_threshold: 0.6

interests:
  - "AI agents"
  - "LLM"

scheduling:
  enabled: false
  cron: "0 7 * * *"

sources:
  - name: "Hugging Face Blog"
    type: rss
    url: "https://huggingface.co/blog/feed.xml"
    enabled: true

  - name: "arXiv"
    type: pdf
    url: "https://arxiv.org"
    enabled: true
    categories: ["cs.AI", "cs.LG"]
    max_items: 10

  - name: "HackerNews"
    type: api
    url: "https://hacker-news.firebaseio.com"
    enabled: true
    max_items: 20
"""


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    """Write a valid config YAML to a temporary file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(VALID_CONFIG, encoding="utf-8")
    return config_file


# ---------------------------------------------------------------------------
# Nominal cases
# ---------------------------------------------------------------------------

def test_load_valid_config(valid_config_file: Path) -> None:
    """Settings object is correctly built from a valid YAML file."""
    settings = load_settings(valid_config_file)

    assert settings.agent.model == "qwen3:8b"
    assert settings.agent.relevance_threshold == 0.6
    assert "AI agents" in settings.interests
    assert len(settings.sources) == 3


def test_sources_by_type_rss(valid_config_file: Path) -> None:
    """sources_by_type returns only RSS sources."""
    settings = load_settings(valid_config_file)
    rss_sources = settings.sources_by_type(SourceType.RSS)

    assert len(rss_sources) == 1
    assert rss_sources[0].name == "Hugging Face Blog"
    assert rss_sources[0].url == "https://huggingface.co/blog/feed.xml"


def test_sources_by_type_pdf(valid_config_file: Path) -> None:
    """sources_by_type returns only PDF sources."""
    settings = load_settings(valid_config_file)
    pdf_sources = settings.sources_by_type(SourceType.PDF)

    assert len(pdf_sources) == 1
    assert pdf_sources[0].name == "arXiv"
    assert pdf_sources[0].categories == ["cs.AI", "cs.LG"]


def test_sources_by_type_excludes_disabled(tmp_path: Path) -> None:
    """sources_by_type excludes disabled sources."""
    config = tmp_path / "config.yaml"
    config.write_text("""
agent:
  model: "qwen3:8b"
interests: ["LLM"]
sources:
  - name: "Active Feed"
    type: rss
    url: "https://example.com/feed.xml"
    enabled: true
  - name: "Disabled Feed"
    type: rss
    url: "https://disabled.com/feed.xml"
    enabled: false
""", encoding="utf-8")

    settings = load_settings(config)
    rss_sources = settings.sources_by_type(SourceType.RSS)

    assert len(rss_sources) == 1
    assert rss_sources[0].name == "Active Feed"


def test_scheduling_defaults(valid_config_file: Path) -> None:
    """Scheduling defaults are applied when not fully specified."""
    settings = load_settings(valid_config_file)

    assert settings.scheduling.enabled is False
    assert settings.scheduling.cron == "0 7 * * *"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_config_file_raises() -> None:
    """FileNotFoundError is raised when config file does not exist."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_settings("/nonexistent/path/config.yaml")


def test_missing_type_field_raises(tmp_path: Path) -> None:
    """ValueError is raised when a source is missing the 'type' field."""
    config = tmp_path / "config.yaml"
    config.write_text("""
agent:
  model: "qwen3:8b"
interests: ["LLM"]
sources:
  - name: "No Type Source"
    url: "https://example.com"
    enabled: true
""", encoding="utf-8")

    with pytest.raises(Exception, match="missing required field 'type'"):
        load_settings(config)


def test_unknown_type_raises(tmp_path: Path) -> None:
    """ValueError is raised when a source has an unknown type."""
    config = tmp_path / "config.yaml"
    config.write_text("""
agent:
  model: "qwen3:8b"
interests: ["LLM"]
sources:
  - name: "Bad Source"
    type: unknown
    url: "https://example.com"
    enabled: true
""", encoding="utf-8")

    with pytest.raises(Exception, match="unknown type"):
        load_settings(config)


def test_all_sources_disabled_raises(tmp_path: Path) -> None:
    """ValidationError is raised when all sources are disabled."""
    config = tmp_path / "config.yaml"
    config.write_text("""
agent:
  model: "qwen3:8b"
interests: ["LLM"]
sources:
  - name: "Disabled"
    type: rss
    url: "https://example.com/feed.xml"
    enabled: false
""", encoding="utf-8")

    with pytest.raises(Exception, match="At least one source must be enabled"):
        load_settings(config)


def test_empty_interests_raises(tmp_path: Path) -> None:
    """ValidationError is raised when interests list is empty."""
    config = tmp_path / "config.yaml"
    config.write_text("""
agent:
  model: "qwen3:8b"
interests: []
sources:
  - name: "Feed"
    type: rss
    url: "https://example.com/feed.xml"
    enabled: true
""", encoding="utf-8")

    with pytest.raises(Exception):
        load_settings(config)


def test_invalid_cron_raises(tmp_path: Path) -> None:
    """ValidationError is raised when cron expression is malformed."""
    config = tmp_path / "config.yaml"
    config.write_text("""
agent:
  model: "qwen3:8b"
interests: ["LLM"]
scheduling:
  enabled: true
  cron: "not a cron"
sources:
  - name: "Feed"
    type: rss
    url: "https://example.com/feed.xml"
    enabled: true
""", encoding="utf-8")

    with pytest.raises(Exception, match="Invalid cron expression"):
        load_settings(config)