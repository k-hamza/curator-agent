"""
Tech Watch Agent — CLI entry point.

Usage:
    # Run the pipeline immediately
    python scripts/run.py --now

    # Start the scheduler (runs daily at the configured cron time)
    python scripts/run.py --schedule

    # Use a custom config file
    python scripts/run.py --now --config path/to/config.yaml

    # Show next scheduled run time without starting
    python scripts/run.py --info
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src/ is on the Python path when running the script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from tech_watch.config.settings import load_settings
from tech_watch.scheduler.runner import run_once, run_scheduled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tech Watch Agent — autonomous technology watch pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run.py --now               Run pipeline immediately
  python scripts/run.py --schedule          Start daily scheduler
  python scripts/run.py --now --config custom.yaml  Use custom config
  python scripts/run.py --info              Show config and next run info
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--now",
        action="store_true",
        help="Run the pipeline immediately",
    )
    mode.add_argument(
        "--schedule",
        action="store_true",
        help="Start the scheduler (blocking — runs until Ctrl+C)",
    )
    mode.add_argument(
        "--info",
        action="store_true",
        help="Display configuration summary and exit",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config.yaml at project root)",
    )

    return parser.parse_args()


def show_info(settings) -> None:
    """Print a summary of the current configuration."""
    print("\n=== Tech Watch Agent — Configuration ===\n")
    print(f"  Model        : {settings.agent.model}")
    print(f"  LLM URL      : {settings.agent.llm_base_url}")
    print(f"  Threshold    : {settings.agent.relevance_threshold}")
    print(f"  Max articles : {settings.agent.max_articles_per_run}")
    print(f"\n  Interests    : {', '.join(settings.interests)}")
    print(f"\n  Sources ({len(settings.sources)}):")
    for source in settings.sources:
        status = "✓" if source.enabled else "✗"
        print(f"    [{status}] {source.name} ({source.type.value})")
    print(f"\n  Scheduling   : {'enabled' if settings.scheduling.enabled else 'disabled'}")
    if settings.scheduling.enabled:
        print(f"  Cron         : {settings.scheduling.cron}")
    print()


def main() -> None:
    args = parse_args()

    # Load and validate configuration
    try:
        settings = load_settings(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)

    if args.info:
        show_info(settings)
        sys.exit(0)

    if args.now:
        asyncio.run(run_once(settings))

    elif args.schedule:
        run_scheduled(settings)


if __name__ == "__main__":
    main()
