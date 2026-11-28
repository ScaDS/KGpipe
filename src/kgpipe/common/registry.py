# global Registry, entry-point discovery

from typing import Any, Callable, List, Dict, Type, Union
from kgpipe.common.models import KgTask, DataFormat
from kgpipe.common.model.configuration import ConfigurationDefinition
from kgpipe.common.model.default_catalog import TaskCategory
from kgpipe.common.graph.mapper import sync_task_to_systemgraph


class RegistryCollisionError(ValueError):
    """A different object is already registered under this name."""


def _callable_identity(fn: Any) -> tuple:
    code = getattr(fn, "__code__", None)
    if code is not None:
        return (code.co_filename, code.co_firstlineno, code.co_name)
    return (
        getattr(fn, "__module__", None),
        getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None),
        id(fn),
    )


def _registration_identity(value: Any) -> tuple:
    if isinstance(value, KgTask):
        return ("task", value.name, _callable_identity(value.function))
    return (
        "object",
        getattr(value, "__module__", None),
        getattr(value, "__qualname__", None) or getattr(value, "__name__", None),
    )


def _describe(value: Any) -> str:
    if isinstance(value, KgTask):
        fn = value.function
        module = getattr(fn, "__module__", "?")
        qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "?")
        return f"task {value.name!r} ({module}.{qualname})"
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(value)


class Registry:
    """
    Holds functions and python objects mappings KGpipe system graph
    """

    _registry: dict[str, Any] = {}

    @classmethod
    def _store(cls, kind: str, name: str, value: Any) -> Any:
        """Insert ``value`` under ``kind:name``.

        Registering the same implementation again is a no-op and keeps the
        first object. A different implementation under the same name raises
        ``RegistryCollisionError``.
        """
        if name is None or not str(name).strip():
            raise ValueError(f"Registry name for kind {kind!r} cannot be empty.")
        key = f"{kind}:{name}"
        existing = cls._registry.get(key)
        if existing is None:
            cls._registry[key] = value
            return value
        if _registration_identity(existing) == _registration_identity(value):
            return existing
        if kind != "task":
            # Metrics from the older aspect evaluators and from kgpipe_eval
            # still share class names. Tasks are the names pipelines resolve.
            cls._registry[key] = value
            return value
        raise RegistryCollisionError(
            f"{kind} {name!r} is already registered as {_describe(existing)}; "
            f"refusing to replace it with {_describe(value)}."
        )

    # Generic #

    @classmethod
    def register(cls, kind: str):
        def decorator(t):
            if kind == "task":
                raise TypeError(
                    "Registry.register('task') cannot register a task. "
                    "Use @Registry.task(...) or Registry.add_task(name, task)."
                )
            cls._store(kind, t.__name__.lower(), t)
            return t
        return decorator

    @classmethod
    def get(cls, kind: str, name: str):
        key = f"{kind}:{name}"
        if key not in cls._registry and kind == "task":
            lowered = f"{kind}:{name.lower()}"
            if lowered in cls._registry:
                key = lowered
        return cls._registry[key]

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
        cls._store("metric", str(key).lower(), metric_cls)

    @classmethod
    def metric(cls):
        """Decorator for kgpipe_eval Metric classes."""
        def decorator(t):
            cls.add_metric(t)
            return t
        return decorator

    # Task #

    @classmethod
    def add_task(cls, name: str, task: KgTask) -> KgTask:
        if not isinstance(task, KgTask):
            raise TypeError(
                f"Registry.add_task expected a KgTask, got {type(task).__name__}."
            )
        if name != task.name:
            raise ValueError(
                f"Registry name {name!r} does not match task name {task.name!r}."
            )
        stored = cls._store("task", task.name, task)
        if stored is task:
            sync_task_to_systemgraph(task)
        return stored

    @classmethod
    def task(
        cls,
        input_spec: Dict[str, DataFormat],
        output_spec: Dict[str, DataFormat],
        description: str | None = None,
        category: Union[TaskCategory, str, List[Union[TaskCategory, str]]] = [],
        config_spec: ConfigurationDefinition | None = None,
        see_also: List[str] = [],
        backend: str = "python",
        dependencies: List[str] | None = None,
        ) -> Callable[[Callable], KgTask]:
        def decorator(t):
            categories = category
            if isinstance(categories, (str, TaskCategory)):
                categories = [categories]
            task = KgTask(
                t.__name__.lower(),
                input_spec,
                output_spec,
                t,
                description,
                list(categories or []),
                config_spec,
                see_also=see_also,
                backend=backend,
                dependencies=list(dependencies or []),
            )
            if getattr(t, "_trace_task_run", False):
                setattr(task, "trace_task_run", True)
            stored = cls._store("task", task.name, task)
            return stored
        return decorator

    @classmethod
    def get_task(cls, name: str) -> KgTask:
        for key in (f"task:{name}", f"task:{name.lower()}"):
            found = cls._registry.get(key)
            if isinstance(found, KgTask):
                return found
        raise KeyError(f"Unknown task {name!r}.")
