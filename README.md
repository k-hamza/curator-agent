# Curator Agent

An autonomous watch agent that collects articles from multiple sources,
filters them using a local LLM, and produces a daily digest in Markdown format.

## Features

- **Multi-source collection** — RSS feeds, HackerNews API, arXiv papers (PDF)
- **LLM-based filtering** — scores each article against your interests using a local model
- **Persistent memory** — never re-reports already processed articles (SQLite)
- **Daily digest** — structured Markdown with overview, summaries, and key points
- **Extensible** — add a new source by creating one file + one YAML entry
- **Fully local** — no cloud API required, runs entirely with Ollama

## Processing flow

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                      │
│              (LangGraph StateGraph)                 │
│        collect, filter, Summary and digest          │
└──────────┬──────────────────────────────────────────┘
           │
    ┌──────▼──────────┐
    │    Collectors   │  (one per source, independants and in parallel)
    ├─────────────────┤
    │  RSSCollector   │  feedparser + httpx async
    │  ArxivCollector │  httpx + pdfplumber
    │  HNCollector    │  official API HackerNews
    └──────┬──────────┘
           │ raw articles (Pydantic models)
    ┌──────▼────────────┐
    │  Deduplication    │  SQLite — filter on seen articles
    │  Capping          │  Keep only max_articles_per_run
    └──────┬────────────┘
           │ new articles
    ┌──────▼────────────┐
    │   FilterAgent     │  LLM scoring (0.0 → 1.0)
    │                   │  Keep articles with score >= threshold
    └──────┬────────────┘
           │
    ┌──────▼────────────┐
    │   SummaryAgent    │  LLM : summarize each article (in its original language)
    └──────┬────────────┘
           │
    ┌──────▼────────────┐
    │   DigestWriter    │  LLM : Overall overview
    │                   │  Grouped by source, sorted by score
    └──────┬────────────┘
           │
    ┌──────▼────────────┐
    │   MarkdownWriter  │  Generate markdown file YYYY-MM-DD_HH-MM.md
    └───────────────────┘
```

## Requirements

- Python 3.11+
- [pyenv](https://github.com/pyenv/pyenv) + pyenv-virtualenv (recommended)
- [Ollama](https://ollama.ai) with at least one model pulled

## Installation

```bash
# Clone the repository
git clone <this_repo>
cd tech-watch-agent

# Create and activate virtual environment
pyenv virtualenv 3.11.9 tech-watch-agent-3.11.9
pyenv local tech-watch-agent-3.11.9

# Install dependencies
pip install -e ".[dev]"
```

## Configuration

Edit `config.yaml` at the project root:

```yaml
agent:
  model: "qwen3:8b"                           # Ollama model to use
  llm_base_url: "http://localhost:11434/v1"   # any OpenAI-compatible endpoint
  relevance_threshold: 0.6                    # min score to include an article

interests:
  - "AI agents"
  - "LLM"
  - "DevOps"
  - "MLOps"

sources:
  - name: "Hugging Face Blog"
    type: rss
    url: "https://huggingface.co/blog/feed.xml"
    enabled: true

  - name: "arXiv"
    type: pdf
    url: "https://arxiv.org"
    enabled: true
    categories: ["cs.AI", "cs.LG", "cs.MA"]
    max_items: 10

  - name: "HackerNews"
    type: api
    url: "https://hacker-news.firebaseio.com"
    enabled: true
    max_items: 20
```

Valid source types: `rss` · `api` · `pdf` · `web` (future)

## Usage

```bash
# Check configuration
python scripts/run.py --info

# Run the pipeline immediately
python scripts/run.py --now

# Start the daily scheduler (blocking)
python scripts/run.py --schedule

# Use a custom config file
python scripts/run.py --now --config path/to/config.yaml
```

Digests are written to `digests/YYYY-MM-DD_HH-MM.md`.

## Project Structure

```
tech-watch-agent/
├── config.yaml                 # user configuration
├── pyproject.toml              # dependencies
├── scripts/
│   └── run.py                  # CLI entry point
├── src/tech_watch/
│   ├── agents/
│   │   ├── digest.py           # DigestWriter — assembles final digest
│   │   ├── filter.py           # FilterAgent — LLM relevance scoring
│   │   └── summary.py          # SummaryAgent — LLM summarization
│   ├── collectors/
│   │   ├── base.py             # BaseCollector + registry
│   │   ├── base_api.py         # BaseApiCollector (HTTP + JSON)
│   │   ├── base_pdf.py         # BasePdfCollector (download + extract)
│   │   ├── arxiv.py            # arXiv collector
│   │   ├── hackernews.py       # HackerNews collector
│   │   └── rss.py              # RSS/Atom collector
│   ├── config/
│   │   └── settings.py         # YAML loading + Pydantic validation
│   ├── graph/
│   │   ├── pipeline.py         # LangGraph StateGraph pipeline
│   │   └── state.py            # GraphState definition
│   ├── llm/
│   │   └── client.py           # OpenAI-compatible LLM client
│   ├── memory/
│   │   └── store.py            # SQLite deduplication store
│   ├── models/
│   │   └── article.py          # Pydantic data contracts
│   ├── output/
│   │   └── markdown.py         # Markdown digest renderer
│   └── scheduler/
│       └── runner.py           # run_once() + run_scheduled()
├── tests/
│   ├── conftest.py
│   ├── test_agents/
│   │   ├── test_digest.py
│   │   ├── test_filter.py
│   │   └── test_summary.py
│   ├── test_collectors/
│   │   ├── test_arxiv.py
│   │   ├── test_hackernews.py
│   │   └── test_rss.py
│   ├── test_config/
│   │   └── test_settings.py
│   ├── test_llm/
│   │   └── test_client.py
│   └── test_memory/
│       └── test_store.py
├── digests/                    # generated digests (git-ignored)
└── data/                       # SQLite + downloaded PDFs (git-ignored)
    └── pdfs/
        └── arXiv/              # one subdirectory per PDF source
```

## Adding a New Source

1. Create `src/tech_watch/collectors/my_source.py`:

```python
from tech_watch.collectors.base import registry
from tech_watch.collectors.base_api import BaseApiCollector
from tech_watch.models.article import RawArticle, SourceType

@registry.register(SourceType.API)
class MySourceCollector(BaseApiCollector):
    async def collect(self, source) -> list[RawArticle]:
        ...
```

2. Add an entry to `config.yaml`:

```yaml
sources:
  - name: "My Source"
    type: api
    url: "https://my-source.com/api"
    enabled: true
    max_items: 20
```

That's it — no other files to modify.

## Running Tests

```bash
# All tests
pytest -v

# Specific module
pytest tests/test_collectors/ -v
pytest tests/test_agents/ -v
```

## Roadmap

### Phase 2
- YouTube channel monitoring with transcript extraction
- Web interface for digest reading (FastAPI + Jinja2)
- Article recommendation and reading order scoring

### Phase 3
- Weekly summary of summaries
- Hot topic detection (spike alerts)
- Alternative export formats (HTML, email)

## Design Decisions

| Decision | Rationale |
|---|---|
| Local LLM via Ollama | No cloud dependency, no cost, privacy |
| LangGraph deterministic pipeline | Fixed collect→filter→summarize→write flow |
| SQLite for memory | Zero dependency, sufficient for URL deduplication |
| Plugin registry for collectors | Add a source = 1 file + 1 YAML entry |
| Pydantic frozen models | Immutable data contracts between pipeline stages |
| No A2A protocol | All agents run in-process; LangGraph state is sufficient |

## License

MIT
