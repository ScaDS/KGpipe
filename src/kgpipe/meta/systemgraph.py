import functools
from uuid import uuid4
from typing import Any, List
from pydantic import BaseModel
from datetime import datetime, timezone

from kgcore.api import KG, BackendName
from kgcore.backend.rdf_rdflib import RDFLibGraph
from kgcore.model.base import KGGraph, KGEntity, KGRelation
from kgcore.common.types import Props, KGId, Lit
from kgpipe.execution.config import load_config

config = load_config()
print(config)
scheme, rest = config.SYS_KG_URL.split("://")
if scheme == "sparql":
    endpoint = f"http://{rest}"
elif scheme == "postgres":
    endpoint = f"postgres://{rest}"
elif scheme == "sqlite":
    endpoint = f"sqlite:///{rest}"
else:
    print("Unsupported schema fallback to memory")
    scheme = "memory"
    # raise ValueError(f"Unsupported scheme: {scheme}")

try:
    SYS_KG: KGGraph = KG(backend=scheme, endpoint=endpoint, username=config.SYS_KG_USR, password=config.SYS_KG_PSW)
except Exception as e:
    print(f"Error creating system graph: {e}")
    SYS_KG = KG(backend="memory")
    print("Using memory backend for system graph")

def kg_class(type: str, description: str = ""):
    """
    Decorator factory: creates a decorator that registers the class
    as a KG entity (type/Class node) once at import time.
    """
    def decorator(cls):
        props = {}
        if description:
            props["description"] = description
        SYS_KG.create_entity([type], id=cls.__name__, props=props)
        return cls

    return decorator


# class Track:
#     """
#     Wrap a class so that each instantiation creates an instance entity in the KG.
#     """
#     def __init__(self, cls):
#         self._cls = cls
#         # Make the wrapper look like the wrapped class (nice for introspection/help)
#         functools.update_wrapper(self, cls)

#     def __call__(self, *args: Any, **kwargs: Any) -> Any:
#         # Construct the actual instance
#         inst = self._cls(*args, **kwargs)

#         # Generate an instance id; replace this with your own scheme if desired
#         inst_id = f"{self._cls.__name__}:{uuid4().hex[:8]}"

#         # Attach the id to the instance for downstream relations (optional)
#         setattr(inst, "_kg_id", inst_id)

#         # Extract props: prefer Pydantic’s model_dump when available
#         if isinstance(inst, BaseModel):
#             props = inst.model_dump()
#         else:
#             # Fallback: best-effort on plain Python objects
#             props = {k: v for k, v in vars(inst).items() if not k.startswith("_")}

#         # Create the instance node typed by the class’ name
#         SYS_KG.create_entity([self._cls.__name__], id=inst_id, props=props)

#         return inst

def Track(_cls=None, *, with_timestamp: bool = False):
    """
    Use as:
        @Track
        @Track(with_timestamp=True)
    """
    def decorator(cls):
        class Tracked(cls):  # subclass the original class
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)

                inst_id = f"{cls.__name__}:{uuid4().hex[:8]}"
                setattr(self, "_kg_id", inst_id)

                if isinstance(self, BaseModel):
                    props = self.model_dump()
                else:
                    props = {k: v for k, v in vars(self).items() if not k.startswith("_")}

                if with_timestamp:
                    props["timestamp"] = datetime.now(timezone.utc).isoformat()

                SYS_KG.create_entity([cls.__name__], id=inst_id, props=props)

        Tracked.__name__ = cls.__name__  # optional cosmetics
        Tracked.__qualname__ = cls.__qualname__
        Tracked.__doc__ = cls.__doc__
        return Tracked

    return decorator if _cls is None else decorator(_cls)

def kg_function(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        call_id = f"{fn.__name__}:{uuid4().hex[:8]}"
        SYS_KG.create_entity(
            ["FunctionCall"],
            id=call_id,
            props={
                "name": fn.__name__,
                # Be careful serializing args/kwargs; this is a toy example:
                "args": repr(args),
                "kwargs": repr(kwargs),
            },
        )
        return result
    return wrapper


# @Track
# @kg_class(type="some",description="Some Class for testing")
# class Some(BaseModel):
#     pred1: str


# @kg_class(type="some")
# class Some2(BaseModel):
#     pred2: List[str]


# @kg_function
# def someFunc():
#     pass
# if __name__ == "__main__":
#     Some(pred1="some")

#     someFunc()
#     print(SYS_KG.asGraph().serialize(format="turtle"))
