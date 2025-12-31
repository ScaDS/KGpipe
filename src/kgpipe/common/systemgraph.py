import functools
from uuid import uuid4
from typing import Any, List
from pydantic import BaseModel
from datetime import datetime, timezone

# from kgcore.api import KG, BackendName

from kgcore.api import KnowledgeGraph, KGEntity, KGRelation, KGProperty, new_id
from kgcore.backend.rdf.rdf_rdflib import RDFLibBackend
from kgcore.backend.rdf.rdf_sparql import RDFSparqlBackend, SparqlAuth
from kgcore.model.rdf.rdf_base import RDFBaseModel

from kgpipe.common.definitions import Task, TaskResult, Pipeline, PipelineResult
from kgpipe.common.config import load_config


config = load_config()
scheme, rest = config.SYS_KG_URL.split("://")

backend = RDFLibBackend()
model = RDFBaseModel()

try:
    if scheme == "sparql":
        print(f"Using SPARQL backend for system graph: {f"http://{rest}"}")
        backend = RDFSparqlBackend(
            endpoint=f"http://{rest}", 
            update_endpoint=f"http://{rest}",
            default_graph="http://kg.org/systemgraph", 
            auth=SparqlAuth(username=config.SYS_KG_USR, password=config.SYS_KG_PSW))
    else:
        print(f"Using RDFLib backend for system graph: {f"http://{rest}"}")
        raise ValueError(f"Unsupported schema: {scheme}")
    SYS_KG: KnowledgeGraph = KnowledgeGraph(model=model, backend=backend)
except Exception as e:
    print(f"Error creating system graph: {e}")
    raise ValueError(f"Error creating system graph: {e}")

def add_task(task: Task):
    SYS_KG.create_entity(id=task.name, types=[task.type], properties={
        "description": task.description,
        "input": task.input,
        "output": task.output,
    })

def add_task_result(task_result: TaskResult):
    SYS_KG.create_entity(id=new_id(),types=["TaskResult"], properties={
        "config": task_result.config,
        "input": task_result.input,
        "output": task_result.output,
    })

def add_pipeline(pipeline: Pipeline):
    SYS_KG.create_entity(id=new_id(),types=["Pipeline"], properties={
        "tasks": pipeline.tasks,
        "input": pipeline.input,
        "output": pipeline.output,
    })

def add_pipeline_result(pipeline_result: PipelineResult):
    SYS_KG.create_entity(id=new_id(),types=["PipelineResult"], properties={
        "task_results": pipeline_result.task_results,
        "eval_results": pipeline_result.eval_results,
        "input": pipeline_result.input,
        "output": pipeline_result.output,
    })

def kg_class(type: str, description: str = ""):
    """
    Decorator factory: creates a decorator that registers the class
    as a KG entity (type/Class node) once at import time.
    """
    def decorator(cls):
        props = []
        if description:
            props.append(KGProperty(key="description", value=description))
        SYS_KG.create_entity(id=cls.__name__, types=[type], properties=None)
        return cls

    return decorator


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
