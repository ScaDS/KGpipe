import functools
from uuid import uuid4
from typing import Any, List, TYPE_CHECKING
from pydantic import BaseModel
from datetime import datetime, timezone

# from kgcore.api import KG, BackendName

from kgcore.api import KnowledgeGraph, KGEntity, KGRelation, KGProperty, new_id
from kgcore.backend.rdf.rdf_rdflib import RDFLibBackend
from kgcore.backend.rdf.rdf_sparql import RDFSparqlBackend, SparqlAuth
from kgcore.model.rdf.rdf_base import RDFBaseModel

from kgpipe.common.definitions import Task, TaskResult, Pipeline, PipelineResult, PipelineRunEntity, ImplementationEntity, MetricEntity, MetricRunEntity
from kgpipe.common.config import load_config
from kgpipe.common.util import encode_string

if TYPE_CHECKING:
    from kgpipe.common.models import KgTask, KgTaskReport


config = load_config()
scheme, rest = config.SYS_KG_URL.split("://")

backend = RDFLibBackend()
model = RDFBaseModel()

try:
    if scheme == "sparql":
        print(f"Using SPARQL backend for system graph: {f"http://{rest}"} with http://kg.org/systemgraph")
        backend = RDFSparqlBackend(
            endpoint=f"http://{rest}", 
            update_endpoint=f"http://{rest}",
            default_graph="http://kg.org/systemgraph", 
            auth=SparqlAuth(username=config.SYS_KG_USR, password=config.SYS_KG_PSW))
    else:
        raise ValueError(f"Unsupported schema: {scheme}")
except Exception as e:
    print(f"Error creating system graph: {e}")
    print(f"Using RDFLib backend for system graph: {f"http://{rest}"}")

SYS_KG: KnowledgeGraph = KnowledgeGraph(model=model, backend=backend)

class PipeKG:

    # cached_implementations: Dict[str, KGEntity] = {}

    @staticmethod
    def add_task(task: "KgTask"):
        from kgpipe.common.models import KgTask  # Import here to avoid circular import
        types = [config.ONTOLOGY_PREFIX+encode_string(c) for c in task.category]
        properties = []
        properties.append(KGProperty(key="description", value=task.description))
        task_entity = SYS_KG.create_entity(id=config.PIPEKG_PREFIX+task.name+"Impl", types=types+[config.ONTOLOGY_PREFIX+"Implementation"], properties=properties)
        for input_name, input_format in task.input_spec.items():
            input_entity = SYS_KG.create_entity(id=config.PIPEKG_PREFIX+task.name+"Impl_"+input_name, types=[config.ONTOLOGY_PREFIX+"Data"], properties={
                "format": input_format,
            })
            SYS_KG.create_relation(type="input", source=task_entity.id, target=input_entity.id)
        for output_name, output_format in task.output_spec.items():
            output_entity = SYS_KG.create_entity(id=config.PIPEKG_PREFIX+task.name+"Impl_"+output_name, types=[config.ONTOLOGY_PREFIX+"Data"], properties={
                "format": output_format,
            })
            SYS_KG.create_relation(type="output", source=task_entity.id, target=output_entity.id)

    def list_tasks(self) -> List["KgTask"]:
        return SYS_KG.list_entities(types=[config.ONTOLOGY_PREFIX+"Implementation"])

    @staticmethod
    def add_task_result(task_result: TaskResult):
        SYS_KG.create_entity(id=new_id(),types=[config.ONTOLOGY_PREFIX+"TaskRun"], properties={
            "task": task_result.task,
            "config": task_result.config,
            "input": task_result.input,
            "output": task_result.output,
            "status": task_result.status,
            "duration": task_result.duration,
        })


    @staticmethod
    def add_task_run(task_run: "KgTaskReport"):
        SYS_KG.create_entity(id=new_id(),types=[config.ONTOLOGY_PREFIX+"TaskReport"], properties={
            "task": task_run.task_name,
            "input": [data.path for data in task_run.inputs],
            "output": [data.path for data in task_run.outputs],
            "status": task_run.status,
            "duration": task_run.duration,
            "error": task_run.error,
        })

    @staticmethod
    def add_pipeline(pipeline: Pipeline):
        SYS_KG.create_entity(id=new_id(),types=["Pipeline"], properties={
            "tasks": pipeline.tasks,
            "input": pipeline.input,
            "output": pipeline.output,
        })

    @staticmethod
    def add_pipeline_result(pipeline_result: PipelineResult):
        SYS_KG.create_entity(id=new_id(),types=["PipelineResult"], properties={
            "task_results": pipeline_result.task_results,
            "eval_results": pipeline_result.eval_results,
            "input": pipeline_result.input,
            "output": pipeline_result.output,
        })

    @staticmethod
    def add_metric(metric: MetricEntity):
        SYS_KG.create_entity(id=config.PIPEKG_PREFIX+encode_string(metric.name),types=[config.ONTOLOGY_PREFIX+"Metric"], properties={
            config.ONTOLOGY_PREFIX+"name": metric.name,
            config.ONTOLOGY_PREFIX+"description": metric.description,
            config.ONTOLOGY_PREFIX+"type": metric.type,
            # "input": metric.input,
            # "output": metric.output,
        })

    @staticmethod
    def add_metric_run(metric_run: MetricRunEntity):
        metric_run_entity = SYS_KG.create_entity(id=new_id(),types=[config.ONTOLOGY_PREFIX+"MetricRun"], properties={
            config.ONTOLOGY_PREFIX+"status": metric_run.status,
            config.ONTOLOGY_PREFIX+"started_at": metric_run.started_at,
            config.ONTOLOGY_PREFIX+"ended_at": metric_run.ended_at,
            config.ONTOLOGY_PREFIX+"value": metric_run.value,
            config.ONTOLOGY_PREFIX+"details": metric_run.details,
            config.ONTOLOGY_PREFIX+"input": metric_run.input[0].uri,
        })
        SYS_KG.create_relation(type=config.ONTOLOGY_PREFIX+"computedMetric", source=metric_run_entity.id, target=metric_run.computedMetric)

    # @staticmethod
    # def find_implementation_by_name(name: str) -> KGEntity:
    #     return SYS_KG.read_entity(id=config.PIPEKG_PREFIX+name, types=[config.ONTOLOGY_PREFIX+"Implementation"])[0]

    @staticmethod
    def add_implementation(implementation: ImplementationEntity):
        SYS_KG.create_entity(id=new_id(),types=[config.ONTOLOGY_PREFIX+"Implementation"], properties={
            "name": implementation.name,
            "usesTool": implementation.usesTool,
            "implementsMethod": implementation.implementsMethod,
            "interface": implementation.interface,

        })

    @staticmethod
    def add_pipeline_run(pipeline_run: PipelineRunEntity):
        pipeline_run_entity = SYS_KG.create_entity(id=new_id(),types=[config.ONTOLOGY_PREFIX+"PipelineRun"], properties={
            "name": pipeline_run.name,
            "status": pipeline_run.status,
            "started_at": pipeline_run.started_at,
            "ended_at": pipeline_run.ended_at
        })
        for idx, task_run in enumerate(pipeline_run.hasTaskRun):
            task_run_entity = SYS_KG.create_entity(id=new_id(),types=[config.ONTOLOGY_PREFIX+"TaskRun"], properties={
                "number": idx,
                "name": task_run.name,
                "status": task_run.status,
                "started_at": task_run.started_at,
                "ended_at": task_run.ended_at,
            })
            SYS_KG.create_relation(type="executesTask", source=task_run_entity.id, target=task_run.executesTask)
            SYS_KG.create_relation(type="usesImplementation", source=task_run_entity.id, target=task_run.usesImplementation)
            SYS_KG.create_relation(type=config.ONTOLOGY_PREFIX+"hasTaskRun", source=pipeline_run_entity.id, target=task_run_entity.id)

        # return pipeline_run_entity


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
