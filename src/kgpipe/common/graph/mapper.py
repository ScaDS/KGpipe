from __future__ import annotations

from kgpipe.common.config import config
from kgpipe.common.graph.systemgraph import PipeKG
from kgpipe.common.model.default_catalog import TaskCategory
from kgpipe.common.util import encode_string

from kgpipe.common.graph.definitions import (
    DataEntity,
    DataEntityId,
    DataSpecEntity,
    DataSpecEntityId,
    DataTypeEntity,
    DataTypeEntityId,
    ImplementationEntity,
    ImplementationEntityId,
    PipelineRunEntity,
    PipelineRunEntityId,
    TaskEntity,
    TaskEntityId,
    TaskRunEntity,
    TaskRunEntityId,
    MetricRunEntity,
    MetricRunEntityId,
    MetricEntity,
    MetricEntityId,
    MeasurementSpecEntity,
    MeasurementSpecEntityId,
    ParameterEntity,
    ParameterEntityId,
    ParameterBindingEntity,
    ParameterBindingEntityId,
    ConfigSpecEntity,
    ConfigSpecEntityId,
    ConfigBindingEntity,
    ConfigBindingEntityId,
    PipelineEntity,
    PipelineEntityId,
    PipelineStepEntity,
)

from typing import TYPE_CHECKING, Any, Type

_SYNCED_TASKS: dict[str, ImplementationEntityId] = {}
_SYNCED_METRICS: dict[str, MetricEntityId] = {}

if TYPE_CHECKING:
    from kgpipe.common.model import (
        DataFormat,
        KgData,
        KgPipe,
        KgPipePlanStep,
        KgTask,
        KgTaskRun,
        KgPipelineRun,
        KgMetricRun,
        ConfigurationDefinition,
        Parameter,
        ConfigurationProfile,
        ParameterBinding,
    )
    from kgpipe.evaluation.base import MetricResult
    from kgpipe_eval.api import Metric as EvalMetric, MeasurementSpec as EvalMeasurementSpec

def task_to_entity(task: "TaskCategory") -> TaskEntityId:
    """Map runtime task definition to a Task entity."""
    name = task
    partOfTask = None
    if isinstance(task, TaskCategory):
        name = task.name
        if task.parent:
            partOfTask = task_to_entity(task.parent)
    task_entity = TaskEntity(
        name=name,
        partOfTask=partOfTask,
    )
    return PipeKG.add_task(task_entity)

def data_type_to_entity(data_type: DataFormat) -> DataTypeEntityId:
    data_type_entity = DataTypeEntity(
        format=data_type,
        data_schema=data_type,
    )
    return PipeKG.add_data_type(data_type_entity)

def data_spec_to_entity(data_spec: tuple[str, DataFormat], implementation_name: str = "") -> DataSpecEntityId:
    data_spec_entity = DataSpecEntity(
        uri=config.PIPEKG_PREFIX + encode_string(implementation_name + "_" + data_spec[0]),
        name=data_spec[0],
        data_type=data_type_to_entity(data_spec[1]),
    )
    return PipeKG.add_data_spec(data_spec_entity)

def data_to_entity(data: "KgData") -> DataEntityId:
    data_entity = DataEntity(
        timestamp=None, # TODO
        version=None, # TODO
        hash=None, # TODO
        size=None, # TODO
        location=data.path.as_uri(),
        data_type=data_type_to_entity(data.format),
    )
    return PipeKG.add_data_entity(data_entity)

def parameter_to_entity(parameter: "Parameter") -> ParameterEntityId:
    datatype = parameter.datatype.value if hasattr(parameter.datatype, "value") else str(parameter.datatype)
    parameter_entity = ParameterEntity(
        key=parameter.name,
        alias_keys=tuple(parameter.native_keys or []),
        datatype=datatype,
        required=bool(parameter.required),
        default_value=parameter.default_value,
        allowed_values=tuple(parameter.allowed_values or []),
        minimum=parameter.minimum,
        maximum=parameter.maximum,
        unit=parameter.unit,
    )
    return PipeKG.add_parameter(parameter_entity)


def config_spec_to_entity(config_spec: "ConfigurationDefinition", implementation_name: str = "") -> ConfigSpecEntityId:
    if config_spec is None:
        return None
    parameter_entities = tuple(
        parameter_to_entity(parameter) for parameter in config_spec.parameters
    )
    config_spec_entity = ConfigSpecEntity(
        name=config_spec.name,
        description=config_spec.description,
        parameters=parameter_entities,
    )
    return PipeKG.add_config_spec(config_spec_entity)

def sync_task_to_systemgraph(task: "KgTask") -> ImplementationEntityId:
    """Register a KgTask in PipeKG if it is not already present.

    If the implementation already exists but has no config_spec while the runtime
    task declares one, attach the missing ConfigSpec relation.
    """
    cached = _SYNCED_TASKS.get(task.name)
    if cached is not None:
        return cached

    if PipeKG.has_implementation(task.name):
        impl_id = ImplementationEntityId(config.PIPEKG_PREFIX + encode_string(task.name))
        if task.config_spec is not None:
            existing = PipeKG.find_implementation(task.name)
            if existing and existing[0].config_spec is None:
                config_spec_id = config_spec_to_entity(task.config_spec, task.name)
                if config_spec_id is not None:
                    from kgpipe.common.graph.systemgraph import SYS_KG
                    from kgpipe.common.graph.definitions import KGPIPE_NS
                    SYS_KG.create_relation(
                        type=KGPIPE_NS.config_spec,
                        source=str(impl_id),
                        target=config_spec_id,
                    )
        _SYNCED_TASKS[task.name] = impl_id
        return impl_id

    impl_id = implementation_to_entity(task)
    _SYNCED_TASKS[task.name] = impl_id
    return impl_id


def implementation_to_entity(implementation: "KgTask") -> ImplementationEntityId:

    input_specs = [data_spec_to_entity(data_spec, implementation.name) for data_spec in implementation.input_spec.items()]

    output_specs = [data_spec_to_entity(data_spec, implementation.name) for data_spec in implementation.output_spec.items()]

    realizes_tasks = [task_to_entity(task) for task in implementation.category]

    config_spec = config_spec_to_entity(implementation.config_spec, implementation.name)

    implementation_entity = ImplementationEntity(
        ### datatype properties ###
        name=implementation.name,
        version="1.0.0", # TODO: get version from implementation
        ### object properties ###
        input_spec=input_specs,
        output_spec=output_specs,
        realizesTask=realizes_tasks,
        usesTool=[], # TODO add usesTool relations
        config_spec=config_spec,
    )
    return PipeKG.add_implementation(implementation_entity)


def _plan_data_spec_to_entity(
    pipeline_name: str,
    step_number: int,
    port_name: str,
    data_type: "DataFormat",
    direction: str,
) -> DataSpecEntityId:
    data_spec_entity = DataSpecEntity(
        uri=config.PIPEKG_PREFIX
        + encode_string(f"{pipeline_name}_{step_number}_{direction}_{port_name}_{data_type}"),
        name=f"{step_number}:{direction}:{port_name}",
        data_type=data_type_to_entity(data_type),
    )
    return PipeKG.add_data_spec(data_spec_entity)


def sync_pipeline_to_systemgraph(pipeline: "KgPipe") -> PipelineEntityId:
    if not pipeline.plan.steps:
        raise ValueError("Pipeline plan is empty. Call build() before syncing to PipeKG.")

    catalog: list[DataSpecEntityId] = []
    if pipeline.plan.source is not None:
        catalog.append(
            _plan_data_spec_to_entity(
                pipeline.name,
                0,
                "source",
                pipeline.plan.source.format,
                "input",
            )
        )
    if pipeline.plan.seed is not None:
        catalog.append(
            _plan_data_spec_to_entity(
                pipeline.name,
                0,
                "seed",
                pipeline.plan.seed.format,
                "input",
            )
        )

    step_ids = []
    pipeline_inputs = list(catalog)
    pipeline_outputs: list[DataSpecEntityId] = []

    for index, plan_step in enumerate(pipeline.plan.steps, start=1):
        task = next((candidate for candidate in pipeline.tasks if candidate.name == plan_step.task), None)
        if task is None:
            raise ValueError(f"Task {plan_step.task} not found in pipeline definition")

        step_inputs: list[DataSpecEntityId] = []
        for input_name, format_spec in task.input_spec.items():
            if input_name == "kg":
                if not catalog:
                    raise ValueError(f"Pipeline {pipeline.name} has no catalog entry for 'kg' input")
                step_inputs.append(catalog[-1])
                continue

            matched = None
            for candidate in catalog:
                port = PipeKG._resolve_data_spec_port(candidate)
                if port is not None and port["format"] == str(format_spec):
                    matched = candidate
                    break
            if matched is None:
                matched = _plan_data_spec_to_entity(
                    pipeline.name,
                    index,
                    input_name,
                    format_spec,
                    "input",
                )
                if matched not in pipeline_inputs:
                    pipeline_inputs.append(matched)
            step_inputs.append(matched)

        step_outputs = [
            _plan_data_spec_to_entity(
                pipeline.name,
                index,
                output_name,
                format_spec,
                "output",
            )
            for output_name, format_spec in task.output_spec.items()
        ]
        if step_outputs:
            catalog = step_outputs + catalog
            pipeline_outputs = step_outputs

        step_task = task_to_entity(task.category[0]) if getattr(task, "category", None) else None
        impl_id = sync_task_to_systemgraph(task)
        step_entity = PipelineStepEntity(
            uri=config.PIPEKG_PREFIX + encode_string(f"{pipeline.name}_step_{index}_{task.name}"),
            name=f"{index}. {task.name}",
            number=index,
            input=step_inputs,
            output=step_outputs,
            stepTask=step_task,
            usesImplementation=impl_id,
        )
        step_ids.append(PipeKG.add_pipeline_step(step_entity))

    pipeline_entity = PipelineEntity(
        uri=config.PIPEKG_PREFIX + encode_string(f"pipeline-{pipeline.name}"),
        name=pipeline.name,
        steps=step_ids,
        firstStep=step_ids[0],
        lastStep=step_ids[-1],
        input=pipeline_inputs,
        output=pipeline_outputs,
    )
    return PipeKG.add_pipeline(pipeline_entity)

def measurement_spec_to_entity(
    measurement: "EvalMeasurementSpec",
    metric_key: str,
) -> MeasurementSpecEntityId:
    alias = tuple(getattr(measurement, "alias", ()) or ())
    measurement_entity = MeasurementSpecEntity(
        uri=config.PIPEKG_PREFIX + encode_string(f"{metric_key}_{measurement.name}"),
        name=measurement.name,
        unit=measurement.unit,
        alias=alias,
    )
    return PipeKG.add_measurement_spec(measurement_entity)


def sync_metric_to_systemgraph(metric: Type[Any] | Any) -> MetricEntityId:
    """Register a kgpipe_eval Metric class in PipeKG if it is not already present."""
    key = getattr(metric, "key", None) or getattr(metric, "__name__", str(metric))
    cached = _SYNCED_METRICS.get(key)
    if cached is not None:
        return cached

    if PipeKG.has_metric(key):
        existing = PipeKG.find_metric(key)
        metric_id = MetricEntityId(config.PIPEKG_PREFIX + encode_string(key))
        declared = list(getattr(metric, "measurements", ()) or ())
        if existing and declared:
            existing_by_name = {
                spec.name: spec
                for spec in PipeKG.resolve_measurement_specs(existing[0].measurements)
            }
            from kgpipe.common.graph.systemgraph import SYS_KG
            from kgpipe.common.graph.definitions import KGPIPE_NS

            for measurement in declared:
                if measurement.name not in existing_by_name:
                    measurement_id = measurement_spec_to_entity(measurement, key)
                    SYS_KG.create_relation(
                        type=KGPIPE_NS.hasMeasurement,
                        source=str(metric_id),
                        target=measurement_id,
                    )
                    continue
                # Refresh aliases on existing measurement specs when declarations change.
                measurement_spec_to_entity(measurement, key)
        _SYNCED_METRICS[key] = metric_id
        return metric_id

    metric_id = metric_to_entity(metric)
    _SYNCED_METRICS[key] = metric_id
    return metric_id


def metric_to_entity(metric: Type[Any] | Any) -> MetricEntityId:
    """Map a kgpipe_eval Metric class (or compatible object) to a Metric entity."""
    key = getattr(metric, "key", None) or getattr(metric, "__name__", str(metric))
    description = getattr(metric, "description", None)
    metric_type = None
    aspect = getattr(metric, "aspect", None)
    if aspect is not None:
        metric_type = getattr(aspect, "value", str(aspect))

    measurement_ids = [
        measurement_spec_to_entity(measurement, key)
        for measurement in (getattr(metric, "measurements", ()) or ())
    ]

    metric_entity = MetricEntity(
        name=key,
        description=description,
        type=metric_type,
        measurements=measurement_ids,
    )
    return PipeKG.add_metric(metric_entity)


def parameter_binding_to_entity(parameter_binding: "ParameterBinding") -> ParameterBindingEntityId:
    parameter_binding_entity = ParameterBindingEntity(
        value=parameter_binding.value,
        parameter=parameter_to_entity(parameter_binding.parameter),
    )
    return PipeKG.add_parameter_binding(parameter_binding_entity)

def config_binding_to_entity(config_profile: "ConfigurationProfile") -> ConfigBindingEntityId:
    config_binding_entity = ConfigBindingEntity(
        name=config_profile.name,
        binding=[parameter_binding_to_entity(binding) for binding in config_profile.bindings],
    )
    return PipeKG.add_config_binding(config_binding_entity)

def task_run_to_entity(task_run: "KgTaskRun") -> TaskRunEntityId:

    input=[data_to_entity(data) for data in task_run.inputs]
    output=[data_to_entity(data) for data in task_run.outputs]
    hasConfigBinding=None # TODO
    usesImplementation=implementation_to_entity(task_run.task)
    hasConfigBinding=config_binding_to_entity(task_run.config_profile) if task_run.config_profile else None

    print(f"hasConfigBinding: {hasConfigBinding}")

    task_run_entity = TaskRunEntity(
        status=task_run.status,
        started_at=task_run.start_ts,
        ended_at=task_run.start_ts + task_run.duration,
        input=input,
        output=output,
        usesImplementation=usesImplementation,
        hasConfigBinding=hasConfigBinding, 
    )
    return PipeKG.add_task_run(task_run_entity) 

def pipeline_run_to_entity(pipeline_run: "KgPipelineRun") -> PipelineRunEntityId:
    pipeline_run_entity = PipelineRunEntity(
        name=pipeline_run.name,
        status=pipeline_run.status,
        started_at=pipeline_run.started_at,
        ended_at=pipeline_run.ended_at,
    )
    return PipeKG.add_pipeline_run(pipeline_run_entity)

# TODO
# def metric_run_to_entity(metric_run: "MetricResult") -> MetricRunEntityId:
#     import time
#     import json
#     computedMetric = metric_to_entity(metric_run.metric)
#     # data_type = data_type_to_entity(DataFormat.ANY)
#     input_entities = [KgData(path=metric_run.kg.path, format=DataFormat.ANY)]
#     input = [data_to_entity(input_entity) for input_entity in input_entities]
#     metric_run_entity = MetricRunEntity(
#         status="success",
#         started_at=time.time(),
#         ended_at=time.time(),
#         computedMetric=computedMetric,
#         input=input,
#         value=metric_run.value, 
#         details=json.dumps(metric_run.details, default=str)
#     )
#     PipeKG.add_metric_run(metric_run_entity)