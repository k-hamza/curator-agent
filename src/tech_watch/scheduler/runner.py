"""
Pipeline runner — manual trigger and APScheduler integration.

Provides two execution modes:
- run_once()    : run the pipeline immediately (manual trigger)
- run_scheduled(): start the APScheduler and run on the configured cron

Usage:
    from tech_watch.scheduler.runner import run_once, run_scheduled
    from tech_watch.config.settings import load_settings

    settings = load_settings()
    await run_once(settings)          # immediate run
    run_scheduled(settings)           # blocking scheduled run
"""

import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from tech_watch.config.settings import Settings
from tech_watch.graph.pipeline import build_pipeline
from tech_watch.memory.store import MemoryStore


async def run_once(settings: Settings) -> dict:
    """
    Run the pipeline immediately and return the final state.

    Args:
        settings: Validated application settings.

    Returns:
        Final GraphState dict after pipeline completion.
    """
    logger.info("Starting pipeline run")

    store = MemoryStore(db_path="data/memory.db")
    store.init()

    pipeline = build_pipeline(settings=settings, store=store)

    try:
        final_state = await pipeline.ainvoke({})

        if final_state.get("completed"):
            logger.success(
                f"Pipeline completed — digest at: {final_state.get('output_path')}"
            )
        else:
            logger.warning("Pipeline completed without generating a digest")

        errors = final_state.get("errors", [])
        if errors:
            logger.warning(f"Non-fatal errors during run: {errors}")

        return final_state

    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {type(e).__name__}: {e}")
        raise


def run_scheduled(settings: Settings) -> None:
    """
    Start the APScheduler and run the pipeline on the configured cron schedule.

    This function blocks until interrupted (Ctrl+C or SIGTERM).

    Args:
        settings: Validated application settings.
                  settings.scheduling.cron defines the cron expression.
                  settings.scheduling.enabled must be True.
    """
    if not settings.scheduling.enabled:
        logger.error(
            "Scheduling is disabled in config.yaml "
            "(scheduling.enabled: false). "
            "Use --now for a manual run."
        )
        return

    cron = settings.scheduling.cron
    logger.info(f"Starting scheduler with cron: '{cron}'")

    scheduler = AsyncIOScheduler()

    async def _scheduled_job() -> None:
        logger.info("Scheduler triggered pipeline run")
        try:
            await run_once(settings)
        except Exception as e:
            logger.error(f"Scheduled run failed: {e}")

    # Parse cron string into APScheduler CronTrigger
    # Cron format: "minute hour day month weekday"
    cron_parts = cron.strip().split()
    trigger = CronTrigger(
        minute=cron_parts[0],
        hour=cron_parts[1],
        day=cron_parts[2],
        month=cron_parts[3],
        day_of_week=cron_parts[4],
    )

    scheduler.add_job(_scheduled_job, trigger=trigger, name="tech-watch-pipeline")
    scheduler.start()

    logger.info(
        f"Scheduler running — next run at: "
        f"{scheduler.get_jobs()[0].next_run_time}"
    )

    # Block until interrupted
    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
        scheduler.shutdown()
