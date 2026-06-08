import functools
import ast
from uuid import uuid4
from typing import Any, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
import hashlib
import json

from kgcore.api import KnowledgeGraph, KGEntity, KGRelation, KGProperty, new_id
from kgcore.backend.rdf.rdf_rdflib import RDFLibBackend
from kgcore.backend.rdf.rdf_sparql import RDFSparqlBackend, SparqlAuth
from kgcore.model.rdf.rdf_base import RDFBaseModel

from kgpipe.common.graph.definitions import (
    KGPIPE_NS,
    ImplementationEntity, ImplementationEntityId, 
    TaskEntity, TaskEntityId, 
    ToolEntity, ToolEntityId,
    DataEntity, DataEntityId, 
    DataSpecEntity, DataSpecEntityId,
    DataTypeEntity, DataTypeEntityId,
    MetricEntity, MetricEntityId,
    MetricRunEntity, MetricRunEntityId,
    TaskRunEntity, TaskRunEntityId,
    ParameterEntity, ParameterEntityId,
    ParameterBindingEntity, ParameterBindingEntityId,
    ConfigSpecEntity, ConfigSpecEntityId,
    ConfigBindingEntity, ConfigBindingEntityId,
)
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
        print(f"Using SPARQL backend for system graph: {f"http://{rest}"} with http://github.com/ScaDS/kgpipe/")
        backend = RDFSparqlBackend(
            endpoint=f"http://{rest}", 
            update_endpoint=f"http://{rest}",
            default_graph="http://github.com/ScaDS/kgpipe/", 
            auth=SparqlAuth(username=config.SYS_KG_USR, password=config.SYS_KG_PSW))
    else:
        raise ValueError(f"Unsupported schema: {scheme}")
except Exception as e:
    print(f"Error creating system graph: {e}")
    print(f"Using RDFLib memory backend for system graph")

SYS_KG: KnowledgeGraph = KnowledgeGraph(model=model, backend=backend)

class PipeKG:
    """
    PipeKG is the system graph for the KGpipe framework. 
    It is a Object Graph Mapper (OGM) for the KGpipe framework.
    It is used to store the entities and relations of the KGpipe framework.
    """

    ### Core Layer Entities ###

    @staticmethod
    @functools.lru_cache
    def add_task(task: TaskEntity) -> TaskEntityId:
        entity_id = config.PIPEKG_PREFIX + encode_string(task.name)
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.Task],
            properties={
                KGPIPE_NS.name: task.name,
                KGPIPE_NS.description: task.description
            },
        )
        if task.partOfTask:
            SYS_KG.create_relation(type=KGPIPE_NS.partOfTask, source=entity_id, target=task.partOfTask)
        return TaskEntityId(entity_id)

    @staticmethod
    @functools.lru_cache
    def add_tool(tool: ToolEntity):
        entity_id = config.PIPEKG_PREFIX + encode_string(tool.name)
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.Tool],
            properties={
                KGPIPE_NS.name: tool.name,
                KGPIPE_NS.homepage: tool.homepage,
            },
        )
        for supports_task in tool.supportsTasks:
            SYS_KG.create_relation(type=KGPIPE_NS.supportsTask, source=entity_id, target=supports_task)
        return ToolEntityId(entity_id)
    
    @staticmethod
    def add_implementation(implementation: ImplementationEntity):
        entity_id = config.PIPEKG_PREFIX + encode_string(implementation.name)
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.Implementation],
            properties={
                KGPIPE_NS.name: implementation.name,
                KGPIPE_NS.version: implementation.version,
            },
        )
        for input_spec in implementation.input_spec:
            SYS_KG.create_relation(type=KGPIPE_NS.input, source=entity_id, target=input_spec)
        for output_spec in implementation.output_spec:
            SYS_KG.create_relation(type=KGPIPE_NS.output, source=entity_id, target=output_spec)
        for realizes_task in implementation.realizesTask:
            SYS_KG.create_relation(type=KGPIPE_NS.realisesTask, source=entity_id, target=realizes_task)
        if implementation.config_spec:
            SYS_KG.create_relation(type=KGPIPE_NS.config_spec, source=entity_id, target=implementation.config_spec)
        return ImplementationEntityId(entity_id)

    @staticmethod
    def find_implementation(
        name: Optional[str] = None,
        # version: Optional[str] = None,
        # input_spec: Optional[List[str]] = None,
        # output_spec: Optional[List[str]] = None,
        # realizes_task: Optional[List[str]] = None,
        # has_parameter: Optional[List[str]] = None,
    ) -> List[ImplementationEntity]:
        entities: List[KGEntity] = SYS_KG.find_entities(
            types=[str(KGPIPE_NS.Implementation)],
        )
        implementations = [ImplementationEntity(
            uri=entity.id,
            name=entity.get_property_value(str(KGPIPE_NS.name))[0],
            version=entity.get_property_value(str(KGPIPE_NS.version))[0],
            input_spec=[DataSpecEntityId(neighbor.id) for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.input))],
            output_spec=[DataSpecEntityId(neighbor.id) for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.output))],
            realizesTask=[TaskEntityId(neighbor.id) for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.realisesTask))],
            # hasParameter=[ParameterEntityId(neighbor.id) for neighbor in entity.get_neighbors(KGPIPE_NS.hasParameter)],
            usesTool=[ToolEntityId(neighbor.id) for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.usesTool))],
            # config_spec=ConfigSpecEntityId(entity.get_property(KGPIPE_NS.config_spec)) if entity.get_property(KGPIPE_NS.config_spec) else None,
        ) for entity in entities]
        if name is not None:
            implementations = [impl for impl in implementations if impl.name == name]
        return implementations

    ### Data Layer Entities ###

    @staticmethod
    @functools.lru_cache
    def add_data_spec(data_spec: DataSpecEntity):
        data_spec_entity = SYS_KG.create_entity(
            id=data_spec.uri if data_spec.uri else new_id(),
            types=[config.ONTOLOGY_PREFIX + "DataSpec"],
            properties={
                config.ONTOLOGY_PREFIX + "name": data_spec.name,
            },
        )
        SYS_KG.create_relation(type=KGPIPE_NS.data_type, source=data_spec_entity.id, target=data_spec.data_type)
        return DataSpecEntityId(data_spec_entity.id)
     
    @staticmethod
    @functools.lru_cache
    def add_data_entity(data_entity: DataEntity):
        entity_id = config.PIPEKG_PREFIX + new_id()
        data_entity_entity = SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.DataEntity],
            properties={}, # TODO
            # properties={
            #     KGPIPE_NS.timestamp: data_entity.timestamp,
            #     KGPIPE_NS.version: data_entity.version,
            #     KGPIPE_NS.hash: data_entity.hash,
            #     KGPIPE_NS.size: data_entity.size,
            # },
        )
        SYS_KG.create_relation(type=KGPIPE_NS.location, source=data_entity_entity.id, target=data_entity.location)
        SYS_KG.create_relation(type=KGPIPE_NS.data_type, source=data_entity_entity.id, target=data_entity.data_type)
        return DataEntityId(data_entity_entity.id)

    @staticmethod
    @functools.lru_cache
    def add_data_type(data_type: DataTypeEntity) -> DataTypeEntityId:
        entity_id = config.PIPEKG_PREFIX + encode_string(data_type.format+"-"+data_type.data_schema)
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.DataType],
            properties={
                KGPIPE_NS.format: data_type.format,
                KGPIPE_NS.schema: data_type.data_schema,
            },
        )
        return DataTypeEntityId(entity_id)

    ### Pipeline Layer Entities ###

    ### Evaluation Layer Entities ###

    def add_metric(metric: MetricEntity):
        pass

    ### Run Layer Entities ###

    def add_task_run(task_run: TaskRunEntity):
        entity_id = config.PIPEKG_PREFIX + new_id()
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.TaskRun],
            properties={
                KGPIPE_NS.status: task_run.status,
                KGPIPE_NS.started_at: task_run.started_at,
                KGPIPE_NS.ended_at: task_run.ended_at,
            },
        )
        for input in task_run.input:
            SYS_KG.create_relation(type=KGPIPE_NS.input, source=entity_id, target=input)
        for output in task_run.output:
            SYS_KG.create_relation(type=KGPIPE_NS.output, source=entity_id, target=output)
        SYS_KG.create_relation(type=KGPIPE_NS.usesImplementation, source=entity_id, target=task_run.usesImplementation)
        return TaskRunEntityId(entity_id)

    def add_metric_run(metric_run: MetricRunEntity):
        pass

    ### Configuration Layer Entities ###

    @staticmethod
    @functools.lru_cache
    def add_parameter(parameter: ParameterEntity):

        payload = json.dumps(parameter.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        stable_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]  # short suffix
        entity_id = config.PIPEKG_PREFIX + encode_string(parameter.key) + "_" + stable_hash
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.Parameter],
            properties={
                KGPIPE_NS.key: parameter.key,
                KGPIPE_NS.alias_keys: parameter.alias_keys,
                KGPIPE_NS.datatype: parameter.datatype,
                KGPIPE_NS.required: parameter.required,
                KGPIPE_NS.default_value: parameter.default_value,
                KGPIPE_NS.allowed_values: parameter.allowed_values,
                # KGPIPE_NS.minimum: parameter.minimum,
                # KGPIPE_NS.maximum: parameter.maximum,
                # KGPIPE_NS.unit: parameter.unit,
            },
        )
        return ParameterEntityId(entity_id)

    def find_parameter(name: str):
        pass
    
    @staticmethod
    def add_parameter_binding(parameter_binding: ParameterBindingEntity):
        payload = json.dumps(parameter_binding.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        stable_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]  # short suffix
        entity_id = parameter_binding.parameter + "_" + stable_hash
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.ParameterBinding],
            properties={
                KGPIPE_NS.value: parameter_binding.value,
            },
        )
        SYS_KG.create_relation(type=KGPIPE_NS.parameter, source=entity_id, target=parameter_binding.parameter)
        return ParameterBindingEntityId(entity_id)

    def find_parameter_binding(name: str):
        pass

    @staticmethod
    @functools.lru_cache
    def add_config_spec(config_spec: ConfigSpecEntity):
        entity_id = config.PIPEKG_PREFIX + encode_string(config_spec.name)
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.ConfigSpec],
            properties={
                KGPIPE_NS.name: config_spec.name,
            },
        )
        for parameter in config_spec.parameters:
            SYS_KG.create_relation(type=KGPIPE_NS.hasParameter, source=entity_id, target=parameter)
        return ConfigSpecEntityId(entity_id)


    def find_config_spec(name: str):
        pass

    @staticmethod
    def add_config_binding(config_binding: ConfigBindingEntity):
        entity_id = config.PIPEKG_PREFIX + encode_string(config_binding.name)
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.ConfigBinding],
            properties={
                KGPIPE_NS.name: config_binding.name,
            },
        )
        for binding in config_binding.binding:
            SYS_KG.create_relation(type=KGPIPE_NS.hasParameterBinding, source=entity_id, target=binding)
        return ConfigBindingEntityId(entity_id)

    def find_config_binding(name: str):
        pass

    ### Utility Functions ###

    @staticmethod
    def sparql_construct(query: str):
        backend : RDFSparqlBackend = SYS_KG.backend
        result = backend.query_sparql(query)
        return result

    @staticmethod
    def _prop_value(properties: List[KGProperty], *keys: str) -> Any:
        """Find a property value by exact key or key suffix."""
        for prop in properties:
            if prop.key in keys:
                return prop.value
        for prop in properties:
            for key in keys:
                if prop.key.endswith(key):
                    return prop.value
        return None

    @staticmethod
    def _to_list(value: Any) -> List[str]:
        """Normalize KG property values to list[str]."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, tuple):
            return [str(v) for v in value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            # Stored literals may contain Python-list string repr.
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return [text]
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            return [text]
        return [str(value)]


