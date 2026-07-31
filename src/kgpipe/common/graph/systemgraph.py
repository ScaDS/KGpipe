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
    PipelineEntity, PipelineStepEntity, PipelineEntityId, PipelineStepEntityId,
    MetricEntity, MetricEntityId,
    MeasurementSpecEntity, MeasurementSpecEntityId,
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
        properties = {
            KGPIPE_NS.name: implementation.name,
            KGPIPE_NS.version: implementation.version,
        }
        if implementation.description:
            properties[KGPIPE_NS.description] = implementation.description
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.Implementation],
            properties=properties,
        )
        for input_spec in implementation.input_spec:
            SYS_KG.create_relation(type=KGPIPE_NS.input, source=entity_id, target=input_spec)
        for output_spec in implementation.output_spec:
            SYS_KG.create_relation(type=KGPIPE_NS.output, source=entity_id, target=output_spec)
        for realizes_task in implementation.realizesTask:
            SYS_KG.create_relation(type=KGPIPE_NS.realisesTask, source=entity_id, target=realizes_task)
        for tool in implementation.usesTool:
            SYS_KG.create_relation(type=KGPIPE_NS.usesTool, source=entity_id, target=tool)
        for see_also in implementation.see_also:
            SYS_KG.create_relation(type=KGPIPE_NS.see_also, source=entity_id, target=see_also)
        if implementation.config_spec:
            SYS_KG.create_relation(type=KGPIPE_NS.config_spec, source=entity_id, target=implementation.config_spec)
        return ImplementationEntityId(entity_id)

    @staticmethod
    def has_implementation(name: str) -> bool:
        """Check whether an implementation with the given name exists."""
        entities = SYS_KG.find_entities(
            types=[str(KGPIPE_NS.Implementation)],
            properties={str(KGPIPE_NS.name): name},
        )
        return len(entities) > 0

    @staticmethod
    def find_implementation(
        name: Optional[str] = None,
        # version: Optional[str] = None,
        # input_spec: Optional[List[str]] = None,
        # output_spec: Optional[List[str]] = None,
        # realizes_task: Optional[List[str]] = None,
        # has_parameter: Optional[List[str]] = None,
    ) -> List[ImplementationEntity]:
        find_kwargs: dict[str, Any] = {"types": [str(KGPIPE_NS.Implementation)]}
        if name is not None:
            find_kwargs["properties"] = {str(KGPIPE_NS.name): name}
        entities: List[KGEntity] = SYS_KG.find_entities(**find_kwargs)
        implementations = []
        for entity in entities:
            config_neighbors = SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.config_spec))
            config_spec_id = (
                ConfigSpecEntityId(config_neighbors[0].id) if config_neighbors else None
            )
            name_vals = entity.get_property_value(str(KGPIPE_NS.name))
            version_vals = entity.get_property_value(str(KGPIPE_NS.version))
            description_vals = entity.get_property_value(str(KGPIPE_NS.description))
            see_also = tuple(
                neighbor.id
                for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.see_also))
            )
            implementations.append(
                ImplementationEntity(
                    uri=entity.id,
                    name=name_vals[0] if name_vals else "",
                    version=version_vals[0] if version_vals else "",
                    description=description_vals[0] if description_vals else None,
                    input_spec=[
                        DataSpecEntityId(neighbor.id)
                        for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.input))
                    ],
                    output_spec=[
                        DataSpecEntityId(neighbor.id)
                        for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.output))
                    ],
                    realizesTask=[
                        TaskEntityId(neighbor.id)
                        for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.realisesTask))
                    ],
                    usesTool=[
                        ToolEntityId(neighbor.id)
                        for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.usesTool))
                    ],
                    config_spec=config_spec_id,
                    see_also=see_also,
                )
            )
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

    @staticmethod
    def add_pipeline_step(step: PipelineStepEntity) -> PipelineStepEntityId:
        entity_id = step.uri or (
            config.PIPEKG_PREFIX + encode_string(f"pipeline-step-{step.name}-{step.number}")
        )
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.PipelineStep],
            properties={
                KGPIPE_NS.name: step.name,
            },
        )
        for input_spec in step.input:
            SYS_KG.create_relation(type=KGPIPE_NS.input, source=entity_id, target=input_spec)
        for output_spec in step.output:
            SYS_KG.create_relation(type=KGPIPE_NS.output, source=entity_id, target=output_spec)
        if step.stepTask is not None:
            SYS_KG.create_relation(type=KGPIPE_NS.stepTask, source=entity_id, target=step.stepTask)
        if step.usesImplementation is not None:
            SYS_KG.create_relation(
                type=KGPIPE_NS.usesImplementation,
                source=entity_id,
                target=step.usesImplementation,
            )
        return PipelineStepEntityId(entity_id)

    @staticmethod
    def add_pipeline(pipeline: PipelineEntity) -> PipelineEntityId:
        entity_id = pipeline.uri or (
            config.PIPEKG_PREFIX + encode_string(f"pipeline-{pipeline.name}")
        )
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.Pipeline],
            properties={
                KGPIPE_NS.name: pipeline.name,
            },
        )
        for step in pipeline.steps:
            SYS_KG.create_relation(type=KGPIPE_NS.hasStep, source=entity_id, target=step)
        for source, target in zip(pipeline.steps, pipeline.steps[1:]):
            SYS_KG.create_relation(type=KGPIPE_NS.nextStep, source=source, target=target)
        for input_spec in pipeline.input:
            SYS_KG.create_relation(type=KGPIPE_NS.input, source=entity_id, target=input_spec)
        for output_spec in pipeline.output:
            SYS_KG.create_relation(type=KGPIPE_NS.output, source=entity_id, target=output_spec)
        return PipelineEntityId(entity_id)

    @staticmethod
    def find_pipeline(name: Optional[str] = None) -> List[PipelineEntity]:
        find_kwargs: dict[str, Any] = {"types": [str(KGPIPE_NS.Pipeline)]}
        if name is not None:
            find_kwargs["properties"] = {str(KGPIPE_NS.name): name}
        entities: List[KGEntity] = SYS_KG.find_entities(**find_kwargs)
        pipelines: List[PipelineEntity] = []
        for entity in entities:
            step_ids = [
                PipelineStepEntityId(neighbor.id)
                for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.hasStep))
            ]
            input_ids = [
                DataSpecEntityId(neighbor.id)
                for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.input))
            ]
            output_ids = [
                DataSpecEntityId(neighbor.id)
                for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.output))
            ]
            name_vals = entity.get_property_value(str(KGPIPE_NS.name))
            pipelines.append(
                PipelineEntity(
                    uri=entity.id,
                    name=name_vals[0] if name_vals else "",
                    steps=step_ids,
                    firstStep=step_ids[0] if step_ids else "",
                    lastStep=step_ids[-1] if step_ids else "",
                    input=input_ids,
                    output=output_ids,
                )
            )
        return pipelines

    ### Evaluation Layer Entities ###

    @staticmethod
    @functools.lru_cache
    def add_measurement_spec(measurement: MeasurementSpecEntity) -> MeasurementSpecEntityId:
        entity_id = measurement.uri or (
            config.PIPEKG_PREFIX + encode_string(f"{measurement.name}_{measurement.unit or 'none'}")
        )
        properties: dict[str, Any] = {
            KGPIPE_NS.name: measurement.name,
        }
        if measurement.unit is not None:
            properties[KGPIPE_NS.unit] = measurement.unit
        if measurement.alias:
            properties[KGPIPE_NS.alias_keys] = list(measurement.alias)
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.MeasurementSpec],
            properties=properties,
        )
        return MeasurementSpecEntityId(entity_id)

    @staticmethod
    def add_metric(metric: MetricEntity) -> MetricEntityId:
        entity_id = config.PIPEKG_PREFIX + encode_string(metric.name)
        properties: dict[str, Any] = {
            KGPIPE_NS.name: metric.name,
        }
        if metric.description is not None:
            properties[KGPIPE_NS.description] = metric.description
        if metric.type is not None:
            properties[KGPIPE_NS.metricType] = metric.type
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.Metric],
            properties=properties,
        )
        for measurement_id in metric.measurements:
            SYS_KG.create_relation(
                type=KGPIPE_NS.hasMeasurement,
                source=entity_id,
                target=measurement_id,
            )
        return MetricEntityId(entity_id)

    @staticmethod
    def has_metric(name: str) -> bool:
        entities = SYS_KG.find_entities(
            types=[str(KGPIPE_NS.Metric)],
            properties={str(KGPIPE_NS.name): name},
        )
        return len(entities) > 0

    @staticmethod
    def find_metric(name: Optional[str] = None) -> List[MetricEntity]:
        find_kwargs: dict[str, Any] = {"types": [str(KGPIPE_NS.Metric)]}
        if name is not None:
            find_kwargs["properties"] = {str(KGPIPE_NS.name): name}
        entities: List[KGEntity] = SYS_KG.find_entities(**find_kwargs)
        metrics: List[MetricEntity] = []
        for entity in entities:
            name_vals = entity.get_property_value(str(KGPIPE_NS.name))
            desc_vals = entity.get_property_value(str(KGPIPE_NS.description))
            type_vals = entity.get_property_value(str(KGPIPE_NS.metricType))
            metrics.append(
                MetricEntity(
                    name=name_vals[0] if name_vals else "",
                    description=desc_vals[0] if desc_vals else None,
                    type=type_vals[0] if type_vals else None,
                    measurements=[
                        MeasurementSpecEntityId(neighbor.id)
                        for neighbor in SYS_KG.get_neighbors(
                            entity.id, str(KGPIPE_NS.hasMeasurement)
                        )
                    ],
                )
            )
        return metrics

    @staticmethod
    def resolve_measurement_specs(
        measurement_ids: List[MeasurementSpecEntityId],
    ) -> List[MeasurementSpecEntity]:
        specs: List[MeasurementSpecEntity] = []
        for measurement_id in measurement_ids:
            entity = SYS_KG.read_entity(str(measurement_id))
            if entity is None:
                continue
            name_vals = entity.get_property_value(str(KGPIPE_NS.name))
            unit_vals = entity.get_property_value(str(KGPIPE_NS.unit))
            alias_vals = entity.get_property_value(str(KGPIPE_NS.alias_keys))
            alias: tuple[str, ...] = ()
            if alias_vals:
                # Property may be stored as a list or as repeated/single values.
                raw = alias_vals if isinstance(alias_vals, list) else list(alias_vals)
                flattened: list[str] = []
                for item in raw:
                    flattened.extend(str(v) for v in PipeKG._to_list(item))
                alias = tuple(flattened)
            specs.append(
                MeasurementSpecEntity(
                    uri=entity.id,
                    name=name_vals[0] if name_vals else "",
                    unit=unit_vals[0] if unit_vals else None,
                    alias=alias,
                )
            )
        return specs

    ### Run Layer Entities ###

    @staticmethod
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

    @staticmethod
    def add_metric_run(metric_run: MetricRunEntity):
        pass

    ### Configuration Layer Entities ###

    @staticmethod
    @functools.lru_cache
    def add_parameter(parameter: ParameterEntity):

        payload = json.dumps(parameter.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        stable_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]  # short suffix
        entity_id = config.PIPEKG_PREFIX + encode_string(parameter.key) + "_" + stable_hash
        properties: dict[str, Any] = {
            KGPIPE_NS.key: parameter.key,
            KGPIPE_NS.alias_keys: list(parameter.alias_keys),
            KGPIPE_NS.datatype: parameter.datatype,
            KGPIPE_NS.required: parameter.required,
            KGPIPE_NS.allowed_values: list(parameter.allowed_values),
        }
        if parameter.default_value is not None:
            properties[KGPIPE_NS.default_value] = parameter.default_value
        if parameter.minimum is not None:
            properties[KGPIPE_NS.minimum] = parameter.minimum
        if parameter.maximum is not None:
            properties[KGPIPE_NS.maximum] = parameter.maximum
        if parameter.unit is not None:
            properties[KGPIPE_NS.unit] = parameter.unit
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.Parameter],
            properties=properties,
        )
        return ParameterEntityId(entity_id)

    @staticmethod
    def find_parameter(
        key: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> List[ParameterEntity]:
        if uri is not None:
            entity = SYS_KG.read_entity(str(uri))
            entities = [entity] if entity is not None else []
        else:
            find_kwargs: dict[str, Any] = {"types": [str(KGPIPE_NS.Parameter)]}
            if key is not None:
                find_kwargs["properties"] = {str(KGPIPE_NS.key): key}
            entities = SYS_KG.find_entities(**find_kwargs)

        parameters: List[ParameterEntity] = []
        for entity in entities:
            if entity is None:
                continue
            parameters.append(PipeKG._parameter_from_entity(entity))
        return parameters

    @staticmethod
    def _parameter_from_entity(entity: KGEntity) -> ParameterEntity:
        props = entity.properties
        datatype_raw = PipeKG._prop_value(props, str(KGPIPE_NS.datatype), "datatype")
        datatype = str(datatype_raw) if datatype_raw is not None else "string"
        if datatype.startswith("ParameterType."):
            datatype = datatype.split(".", 1)[1]

        key_raw = PipeKG._prop_value(props, str(KGPIPE_NS.key), "key")
        required_raw = PipeKG._prop_value(props, str(KGPIPE_NS.required), "required")
        default_raw = PipeKG._prop_value(props, str(KGPIPE_NS.default_value), "default_value")
        alias_raw = PipeKG._prop_value(props, str(KGPIPE_NS.alias_keys), "alias_keys")
        allowed_raw = PipeKG._prop_value(props, str(KGPIPE_NS.allowed_values), "allowed_values")
        minimum_raw = PipeKG._prop_value(props, str(KGPIPE_NS.minimum), "minimum")
        maximum_raw = PipeKG._prop_value(props, str(KGPIPE_NS.maximum), "maximum")
        unit_raw = PipeKG._prop_value(props, str(KGPIPE_NS.unit), "unit")

        return ParameterEntity(
            uri=entity.id,
            key=str(key_raw) if key_raw is not None else "",
            alias_keys=tuple(PipeKG._to_list(alias_raw)),
            datatype=datatype,
            required=PipeKG._to_bool(required_raw),
            default_value=PipeKG._coerce_scalar(default_raw),
            allowed_values=tuple(
                v
                for v in (PipeKG._coerce_scalar(item) for item in PipeKG._to_list(allowed_raw))
                if v is not None
            ),
            minimum=PipeKG._to_float(minimum_raw),
            maximum=PipeKG._to_float(maximum_raw),
            unit=str(unit_raw) if unit_raw is not None else None,
        )
    
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
        properties: dict[str, Any] = {
            KGPIPE_NS.name: config_spec.name,
        }
        if config_spec.description is not None:
            properties[KGPIPE_NS.description] = config_spec.description
        SYS_KG.create_entity(
            id=entity_id,
            types=[KGPIPE_NS.ConfigSpec],
            properties=properties,
        )
        for parameter in config_spec.parameters:
            SYS_KG.create_relation(type=KGPIPE_NS.hasParameter, source=entity_id, target=parameter)
        return ConfigSpecEntityId(entity_id)

    @staticmethod
    def find_config_spec(
        name: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> List[ConfigSpecEntity]:
        if uri is not None:
            entity = SYS_KG.read_entity(str(uri))
            entities = [entity] if entity is not None else []
        else:
            find_kwargs: dict[str, Any] = {"types": [str(KGPIPE_NS.ConfigSpec)]}
            if name is not None:
                find_kwargs["properties"] = {str(KGPIPE_NS.name): name}
            entities = SYS_KG.find_entities(**find_kwargs)

        specs: List[ConfigSpecEntity] = []
        for entity in entities:
            if entity is None:
                continue
            name_vals = entity.get_property_value(str(KGPIPE_NS.name))
            description = PipeKG._prop_value(
                entity.properties, str(KGPIPE_NS.description), "description"
            )
            param_ids = tuple(
                ParameterEntityId(neighbor.id)
                for neighbor in SYS_KG.get_neighbors(entity.id, str(KGPIPE_NS.hasParameter))
            )
            specs.append(
                ConfigSpecEntity(
                    uri=entity.id,
                    name=name_vals[0] if name_vals else "",
                    description=str(description) if description is not None else None,
                    parameters=param_ids,
                )
            )
        return specs

    @staticmethod
    def resolve_config_spec(
        config_spec_id: Optional[ConfigSpecEntityId],
    ) -> Optional[ConfigSpecEntity]:
        """Resolve a ConfigSpec id to a ConfigSpecEntity with ParameterEntity neighbors expanded.

        Returns a ConfigSpecEntity whose ``parameters`` field still holds ParameterEntityIds;
        use :meth:`resolve_config_spec_parameters` for fully materialised ParameterEntity objects.
        """
        if config_spec_id is None:
            return None
        specs = PipeKG.find_config_spec(uri=str(config_spec_id))
        return specs[0] if specs else None

    @staticmethod
    def resolve_config_spec_parameters(
        config_spec_id: Optional[ConfigSpecEntityId],
    ) -> tuple[Optional[ConfigSpecEntity], List[ParameterEntity]]:
        """Resolve config spec and its Parameter entities."""
        spec = PipeKG.resolve_config_spec(config_spec_id)
        if spec is None:
            return None, []
        parameters: List[ParameterEntity] = []
        for param_id in spec.parameters:
            found = PipeKG.find_parameter(uri=str(param_id))
            if found:
                parameters.append(found[0])
        return spec, parameters

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
    def _to_list(value: Any) -> List[Any]:
        """Normalize KG property values to a list (preserving scalar types when possible)."""
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
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
                    return list(parsed)
            return [text]
        return [value]

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no", ""}:
            return False
        return bool(value)

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_scalar(value: Any) -> Optional[str | int | float | bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            lowered = text.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            try:
                if "." in text or "e" in lowered:
                    return float(text)
                return int(text)
            except ValueError:
                return text
        return str(value)

    @staticmethod
    def resolve_data_spec_formats(data_spec_ids: List[DataSpecEntityId]) -> List[str]:
        """Resolve DataSpec entity IDs to format strings (unique, sorted).

        Prefer :meth:`resolve_data_spec_ports` when multiplicity / port names matter.
        """
        formats: list[str] = []
        seen: set[str] = set()
        for port in PipeKG.resolve_data_spec_ports(data_spec_ids):
            fmt = port["format"]
            if fmt not in seen:
                seen.add(fmt)
                formats.append(fmt)
        return sorted(formats)

    @staticmethod
    def resolve_data_spec_ports(data_spec_ids: List[DataSpecEntityId]) -> List[dict[str, str]]:
        """Resolve DataSpecs to named ports ``{name, format}`` (preserves multiplicity)."""
        ports: list[dict[str, str]] = []
        for data_spec_id in data_spec_ids:
            port = PipeKG._resolve_data_spec_port(data_spec_id)
            if port is not None:
                ports.append(port)
        return ports

    @staticmethod
    def _resolve_data_spec_port(data_spec_id: DataSpecEntityId) -> Optional[dict[str, str]]:
        entity = SYS_KG.read_entity(str(data_spec_id))
        if entity is None:
            return None
        name = PipeKG._prop_value(
            entity.properties,
            str(KGPIPE_NS.name),
            config.ONTOLOGY_PREFIX + "name",
            "name",
        )
        fmt = PipeKG._resolve_data_spec_format(data_spec_id)
        if fmt is None:
            return None
        port_name = str(name) if name is not None else fmt
        return {"name": port_name, "format": fmt}

    @staticmethod
    def _resolve_data_spec_format(data_spec_id: DataSpecEntityId) -> Optional[str]:
        data_types = SYS_KG.get_neighbors(str(data_spec_id), str(KGPIPE_NS.data_type))
        if not data_types:
            return None
        fmt = PipeKG._prop_value(data_types[0].properties, str(KGPIPE_NS.format), "format")
        if fmt is None:
            fmt = PipeKG._prop_value(data_types[0].properties, str(KGPIPE_NS.schema), "schema")
        return str(fmt) if fmt is not None else None


