"""
Base collector interface and collector registry.

Every collector must:
1. Inherit from BaseCollector
2. Register itself with @registry.register("type_name")
3. Implement the collect() method

Usage:
    # In a collector module (e.g. rss.py):
    from tech_watch.collectors.base import BaseCollector, registry

    @registry.register(SourceType.RSS)
    class RSSCollector(BaseCollector):
        async def collect(self, source: BaseSourceSettings) -> list[RawArticle]:
            ...

    # In the pipeline:
    from tech_watch.collectors.base import registry

    collectors = registry.get_enabled(settings)
    for source, collector in collectors:
        articles = await collector.collect(source)
"""

import importlib
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from tech_watch.models.article import RawArticle, SourceType
from tech_watch.config.settings import BaseSourceSettings, Settings

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Base collector interface
# ---------------------------------------------------------------------------

class BaseCollector(ABC):
    """
    Abstract base class for all collectors.

    Each collector is responsible for fetching raw articles from a single
    source type. It receives a validated source configuration and returns
    a list of RawArticle instances.

    Collectors must be stateless — all state lives in the pipeline graph.
    """

    @abstractmethod
    async def collect(self, source: BaseSourceSettings) -> list[RawArticle]:
        """
        Fetch articles from the given source.

        Args:
            source: Validated source configuration from config.yaml.

        Returns:
            List of RawArticle instances. Empty list if nothing was fetched.
            Never raises — errors are logged and an empty list is returned.
        """
        ...

    async def safe_collect(self, source: BaseSourceSettings) -> list[RawArticle]:
        """
        Wrapper around collect() that catches all exceptions.
        Called by the pipeline — never call collect() directly from the graph.

        Returns an empty list on failure so the pipeline can continue
        with other sources.
        """
        try:
            articles = await self.collect(source)
            logger.info(
                f"[{source.name}] collected {len(articles)} article(s)"
            )
            return articles
        except Exception as e:
            logger.error(
                f"[{source.name}] collection failed: {type(e).__name__}: {e}"
            )
            return []


# ---------------------------------------------------------------------------
# Collector registry
# ---------------------------------------------------------------------------

class CollectorRegistry:
    """
    Registry that maps SourceType values to collector classes.

    Collectors self-register using the @registry.register decorator.
    The pipeline queries the registry at runtime to get the right
    collector for each enabled source.
    """

    def __init__(self) -> None:
        self._registry: dict[SourceType, type[BaseCollector]] = {}

    def register(self, source_type: SourceType):
        """
        Decorator that registers a collector class for a given SourceType.

        Usage:
            @registry.register(SourceType.RSS)
            class RSSCollector(BaseCollector):
                ...
        """
        def decorator(cls: type[BaseCollector]) -> type[BaseCollector]:
            if not issubclass(cls, BaseCollector):
                raise TypeError(
                    f"Cannot register {cls.__name__}: "
                    f"must be a subclass of BaseCollector"
                )
            if source_type in self._registry:
                logger.warning(
                    f"Overriding existing collector for '{source_type.value}' "
                    f"with {cls.__name__}"
                )
            self._registry[source_type] = cls
            logger.debug(
                f"Registered collector {cls.__name__} for type '{source_type.value}'"
            )
            return cls
        return decorator

    def get(self, source_type: SourceType) -> type[BaseCollector] | None:
        """Return the collector class for a given SourceType, or None."""
        return self._registry.get(source_type)

    def get_enabled(
        self, settings: Settings
    ) -> list[tuple[BaseSourceSettings, BaseCollector]]:
        """
        Return instantiated collectors for all enabled sources.

        For each enabled source in settings, looks up the registered
        collector class and returns a (source_config, collector_instance) pair.

        Collectors that accept a Settings argument (e.g. ArxivCollector)
        receive it automatically at instantiation. Collectors with no
        constructor arguments (e.g. RSSCollector) are instantiated normally.

        Sources with no registered collector are logged as warnings and skipped.

        Args:
            settings: Validated application settings.

        Returns:
            List of (source_config, collector_instance) pairs, ready to run.
        """
        import inspect

        result = []
        for source in settings.sources:
            if not source.enabled:
                continue

            collector_cls = self._registry.get(source.type)
            if collector_cls is None:
                logger.warning(
                    f"No collector registered for type '{source.type.value}' "
                    f"(source: '{source.name}') — skipping"
                )
                continue

            # Pass settings if the collector's __init__ accepts it
            sig = inspect.signature(collector_cls.__init__)
            if "settings" in sig.parameters:
                collector = collector_cls(settings=settings)
            else:
                collector = collector_cls()

            result.append((source, collector))

        return result

    def registered_types(self) -> list[SourceType]:
        """Return all currently registered SourceType values."""
        return list(self._registry.keys())


# ---------------------------------------------------------------------------
# Global registry instance
# ---------------------------------------------------------------------------

# Single registry instance shared across the entire application.
# Imported by each collector module to self-register.
registry = CollectorRegistry()


# ---------------------------------------------------------------------------
# Auto-discovery — import all collector modules so they self-register
# ---------------------------------------------------------------------------

def autodiscover_collectors() -> None:
    """
    Import all modules in the collectors package so their @registry.register
    decorators execute and populate the global registry.

    Called once at application startup (in pipeline.py or run.py).
    Safe to call multiple times — re-importing already-loaded modules is a no-op.
    """
    collectors_path = Path(__file__).parent
    package_name = __name__.rsplit(".", 1)[0]  # "tech_watch.collectors"

    for module_info in pkgutil.iter_modules([str(collectors_path)]):
        if module_info.name == "base":
            continue  # skip this module itself

        module_name = f"{package_name}.{module_info.name}"
        try:
            importlib.import_module(module_name)
            logger.debug(f"Auto-discovered collector module: {module_name}")
        except ImportError as e:
            logger.error(f"Failed to import collector module '{module_name}': {e}")
