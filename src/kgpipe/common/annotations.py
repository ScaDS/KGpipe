from rdflib import OWL, RDFS
from kgpipe.common.graph.systemgraph import SYS_KG, PipeKG
from kgcore.api import KGProperty
from typing import get_origin, get_args, Union

# TODO move to kgcore.api 
# TODO handle resolution of SYS_KG
def kg_class(description: str = ""):
    """
    Decorator factory: creates a decorator that registers the class
    as a KG entity (type/Class node) once at import time.
    """
    def decorator(cls):
        # print("kg_class decorator called for class: ", cls.__name__)
        # add owl class
        props = []
        if description:
            props.append(KGProperty(key="description", value=description))
        class_et = SYS_KG.create_entity(id=cls.__name__, types=[OWL.Class], properties=props)
        # get variables of the class
        # for each variable, add a DatatypeProperty or ObjectProperty
        for var in cls.__annotations__.keys():
            annotation = cls.__annotations__[var]
            is_object_property = False
            with_object_type = None
            
            # Check if it's a direct class type (e.g., Data)
            if isinstance(annotation, type) and annotation not in (str, int, float, bool):
                is_object_property = True
                with_object_type = annotation.__name__
            else:
                # Check for generic types (List, Dict, Optional, etc.)
                origin = get_origin(annotation)
                args = get_args(annotation)
                
                # Check if any of the type arguments are classes (not primitives)
                # Note: List[str], List[int], etc. remain DatatypeProperty because
                # str, int, float, bool are primitive types, not class types
                for arg in args:
                    if isinstance(arg, type):
                        # Skip primitive types - these result in DatatypeProperty
                        # Only non-primitive class types result in ObjectProperty
                        if arg not in (str, int, float, bool, type(None)):
                            is_object_property = True
                            with_object_type = arg.__name__
                            break
                    # Handle string forward references (e.g., "Data" in quotes)
                    elif isinstance(arg, str):
                        # Try to resolve the string reference
                        # For now, assume string references might be classes
                        is_object_property = True
                        break
            
            # Also check if it's a class attribute (old behavior for dataclasses)
            if not is_object_property:
                attr_value = getattr(cls, var, None)
                if attr_value is not None and isinstance(attr_value, type) and attr_value not in (str, int, float, bool):
                    is_object_property = True
            
            if is_object_property:
                # add ObjectProperty
                prop_et = SYS_KG.create_entity(id=cls.__name__+"_"+var, types=[OWL.ObjectProperty], properties={
                    RDFS.label: var,
                })
                SYS_KG.create_relation(source=prop_et.id, target=class_et.id, type=str(RDFS.domain))
                if with_object_type:
                    SYS_KG.create_relation(source=prop_et.id, target=with_object_type, type=str(RDFS.range))
            else:
                # add DatatypeProperties
                prop_et = SYS_KG.create_entity(id=cls.__name__+"_"+var, types=[OWL.DatatypeProperty], properties={
                    RDFS.label: var,
                })
                SYS_KG.create_relation(source=prop_et.id, target=class_et.id, type=str(RDFS.domain))
        return cls

    return decorator

 

def trace_metric_run(): pass


def trace_task_run(obj):
    """
    Mark a task (function or `KgTask`) so that its `.run()` persists a TaskRun in `PipeKG`.

    Works with either decorator order:

    ```python
    @trace_task_run
    @Registry.task(...)
    def my_task(...): ...

    # or
    @Registry.task(...)
    @trace_task_run
    def my_task(...): ...
    ```
    """
    setattr(obj, "trace_task_run", True)
    # TODO use logger print(f"trace_task_run decorator called for object: {obj.__name__}")
    return obj

def trace_pipeline_run(obj):
    """
    Mark a pipeline (function or `KgPipeline`) so that its `.run()` persists a PipelineRun in `PipeKG`.
    """
    setattr(obj, "trace_pipeline_run", True)
    # TODO use logger print(f"trace_pipeline_run decorator called for object: {obj.__name__}")
    return obj


# def Track(_cls=None, *, with_timestamp: bool = False):
#     """
#     Use as:
#         @Track
#         @Track(with_timestamp=True)
#     """
#     def decorator(cls):
#         class Tracked(cls):  # subclass the original class
#             def __init__(self, *args: Any, **kwargs: Any):
#                 super().__init__(*args, **kwargs)

#                 inst_id = f"{cls.__name__}:{uuid4().hex[:8]}"
#                 setattr(self, "_kg_id", inst_id)

#                 if isinstance(self, BaseModel):
#                     props = self.model_dump()
#                 else:
#                     props = {k: v for k, v in vars(self).items() if not k.startswith("_")}

#                 if with_timestamp:
#                     props["timestamp"] = datetime.now(timezone.utc).isoformat()

#                 SYS_KG.create_entity([cls.__name__], id=inst_id, props=props)

#         Tracked.__name__ = cls.__name__  # optional cosmetics
#         Tracked.__qualname__ = cls.__qualname__
#         Tracked.__doc__ = cls.__doc__
#         return Tracked

#     return decorator if _cls is None else decorator(_cls)

# def kg_function(fn):
#     @functools.wraps(fn)
#     def wrapper(*args, **kwargs):
#         result = fn(*args, **kwargs)
#         call_id = f"{fn.__name__}:{uuid4().hex[:8]}"
#         SYS_KG.create_entity(
#             ["FunctionCall"],
#             id=call_id,
#             props={
#                 "name": fn.__name__,
#                 # Be careful serializing args/kwargs; this is a toy example:
#                 "args": repr(args),
#                 "kwargs": repr(kwargs),
#             },
#         )
#         return result
#     return wrapper
