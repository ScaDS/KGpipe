 # global Registry, entry-point discovery
 
from typing import Any, Callable
from kgpipe.common.models import KgTask, DataFormat

# TODO add also to system graph



class Registry:
    """
    Holds functions and python objects
    """

    _registry: dict[str, Any] = {}

    @classmethod
    def register(cls, kind: str):
        def decorator(t):
            cls._registry[f"{kind}:{t.__name__.lower()}"] = t
            return t
        return decorator

    @classmethod
    def metric(cls):
        def decorator(t):
            cls._registry[f"metric:{t.__name__.lower()}"] = t
            return t
        return decorator

    @classmethod
    def task(
        cls, 
        input_spec: dict[str, DataFormat], 
        output_spec: dict[str, DataFormat], 
        description: str | None = None, 
        category: list[str] = []
        ) -> Callable[[Callable], KgTask]:
        def decorator(t):
            task = KgTask(t.__name__.lower(), input_spec, output_spec, t, description, category)
            cls._registry[f"task:{t.__name__.lower()}"] = task
            return task
        return decorator

    @classmethod
    def get(cls, kind: str, name: str):
        return cls._registry[f"{kind}:{name}"]
    
    @classmethod
    def get_task(cls, name: str) -> KgTask:
        return cls._registry[f"task:{name}"]

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