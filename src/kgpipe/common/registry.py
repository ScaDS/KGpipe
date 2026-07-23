 # global Registry, entry-point discovery
 
from typing import Any, Callable, List, Dict, Type
from kgpipe.common.models import KgTask, DataFormat
# from kgpipe.common.graph.systemgraph import PipeKG
from kgpipe.common.graph.definitions import MetricEntity, TaskEntity
from kgpipe.common.model.configuration import ConfigurationDefinition
from kgpipe.common.graph.mapper import sync_task_to_systemgraph

class Registry:
    """
    Holds functions and python objects mappings KGpipe system graph
    """

    _registry: dict[str, Any] = {}

    # Generic #

    @classmethod
    def register(cls, kind: str):
        def decorator(t):
            cls._registry[f"{kind}:{t.__name__.lower()}"] = t
            return t
        return decorator

    @classmethod
    def get(cls, kind: str, name: str):
        return cls._registry[f"{kind}:{name}"]

    @classmethod
    def list(cls, kind: str):
        """List all registered items of a specific kind."""
        items = []
        for key, value in cls._registry.items():
            if key.startswith(f"{kind}:"):
                items.append(value)
        return items

    @classmethod
    def list_all(cls):
        return cls._registry

    # Metric #

    @classmethod
    def add_metric(cls, metric_cls: Type[Any]):
        """Register a kgpipe_eval Metric class by its `key`."""
        key = getattr(metric_cls, "key", None) or metric_cls.__name__
        cls._registry[f"metric:{str(key).lower()}"] = metric_cls

    @classmethod
    def metric(cls):
        """Decorator for kgpipe_eval Metric classes."""
        def decorator(t):
            cls.add_metric(t)
            return t
        return decorator

    # Task #

    @classmethod
    def add_task(cls, name: str, task: KgTask):
        cls._registry[f"task:{task.name}"] = task
        sync_task_to_systemgraph(task)

    @classmethod
    def task(
        cls, 
        input_spec: Dict[str, DataFormat], 
        output_spec: Dict[str, DataFormat], 
        description: str | None = None, 
        category: List[str] = [],
        config_spec: ConfigurationDefinition | None = None
        ) -> Callable[[Callable], KgTask]:
        def decorator(t):
            task = KgTask(t.__name__.lower(), input_spec, output_spec, t, description, category, config_spec)
            if getattr(t, "_trace_task_run", False):
                setattr(task, "trace_task_run", True)
            cls._registry[f"task:{t.__name__.lower()}"] = task
            return task
        return decorator

    @classmethod
    def get_task(cls, name: str) -> KgTask:
        return cls._registry[f"task:{name}"]
