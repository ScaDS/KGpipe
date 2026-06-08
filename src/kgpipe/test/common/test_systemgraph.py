from uuid import uuid4

from kgpipe.common.definitions import (
    DataHandle,
    ImplementationEntity,
    MethodEntity,
    MetricEntity,
    PipelineEntity,
    TaskRunEntity,
    ToolEntity,
)
from kgpipe.common.systemgraph import PipeKG


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def test_core_layer_method_tool_and_implementation():
    method_name = _uid("method")
    tool_name = _uid("tool")
    impl_name = _uid("impl")

    method = MethodEntity(name=method_name, realizesTask=["task:a"])
    tool = ToolEntity(name=tool_name, providesMethods=["method:a"])
    implementation = ImplementationEntity(
        name=impl_name,
        input_spec=["text/csv"],
        output_spec=["application/json"],
        implementsMethod=["method:a"],
        hasParameter=["param:a"],
        usesTool=["tool:a"],
    )

    PipeKG.add_method(method)
    PipeKG.add_tool(tool)
    PipeKG.add_implementation(implementation)

    found_method = PipeKG.find_method(method_name)
    found_tool = PipeKG.find_tool(tool_name)
    found_implementation = PipeKG.find_implementation(impl_name)

    assert found_method is not None
    assert found_method.name == method_name
    assert "task:a" in found_method.realizesTask

    assert found_tool is not None
    assert found_tool.name == tool_name
    assert "method:a" in found_tool.providesMethods

    assert found_implementation is not None
    assert found_implementation.name == impl_name
    assert found_implementation.input_spec == ["text/csv"]
    assert found_implementation.output_spec == ["application/json"]


def test_data_layer_artifact_type_and_spec():
    artifact_uri = f"file:///{_uid('artifact')}.csv"
    artifact_type = _uid("artifact_type")
    spec_name = _uid("spec")
    specification = '{"type":"object","properties":{"name":{"type":"string"}}}'
    data = DataHandle(
        uri=artifact_uri,
        type="text/csv",
        version="1.0.0",
        hash="abc123",
        size=42,
    )

    PipeKG.add_data_artifact(data)
    PipeKG.add_data_artifact_type(artifact_type)
    PipeKG.add_data_artifact_spec(spec_name, specification)

    found_data = PipeKG.find_data_artifact(artifact_uri)
    found_type = PipeKG.find_data_artifact_type(artifact_type)
    found_spec = PipeKG.find_data_artifact_spec(spec_name)

    assert found_data is not None
    assert found_data.uri == artifact_uri
    assert found_data.type == "text/csv"
    assert found_data.version == "1.0.0"
    assert found_type == artifact_type
    assert found_spec == specification


def test_pipeline_layer_pipeline_step_and_definition():
    pipeline_name = _uid("pipeline")
    step_task = "task:clean"
    definition_name = _uid("pipeline_def")
    pipeline_id = f"pipeline:{pipeline_name}"

    pipeline = PipelineEntity(name=pipeline_name, tasks=[step_task], input=[], output=[])
    PipeKG.add_pipeline(pipeline)
    PipeKG.add_pipeline_step(pipeline_name=pipeline_name, step_number=1, task_id=step_task)
    PipeKG.add_pipeline_definition(name=definition_name, pipeline_id=pipeline_id)

    found_pipeline = PipeKG.find_pipeline(pipeline_name)
    found_step = PipeKG.find_pipeline_step(pipeline_name, 1)
    found_definition = PipeKG.find_pipeline_definition(definition_name)

    assert found_pipeline is not None
    assert found_pipeline.name == pipeline_name
    assert step_task in found_pipeline.tasks
    assert found_step is not None
    assert found_definition is not None


def test_metrics_layer_add_and_find_metric():
    metric_name = _uid("metric")
    metric = MetricEntity(name=metric_name, description="Accuracy metric", type="score")
    PipeKG.add_metric(metric)

    found_metric = PipeKG.find_metric(metric_name)

    assert found_metric is not None
    assert found_metric.name == metric_name
    assert found_metric.description == "Accuracy metric"
    assert found_metric.type == "score"


def test_run_layer_add_task_run():
    task_run = TaskRunEntity(
        number=1,
        name=_uid("task_run"),
        status="success",
        started_at=1.0,
        ended_at=2.0,
        input=[DataHandle(uri="file:///in.csv", type="text/csv")],
        output=[DataHandle(uri="file:///out.csv", type="text/csv")],
        executesTask="task:clean",
        usesImplementation="impl:clean_v1",
        hasParameterBinding=[],
    )

    PipeKG.add_task_run(task_run)