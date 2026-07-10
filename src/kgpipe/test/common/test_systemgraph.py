from uuid import uuid4

from kgpipe.common.discovery import discover_entry_points, get_registered_tasks
from kgpipe.common.graph.definitions import (
    DataEntity,
    DataSpecEntity,
    DataTypeEntity,
    ImplementationEntity,
    MetricEntity,
    TaskEntity,
    TaskRunEntity,
    ToolEntity,
)
from kgpipe.common.graph.systemgraph import PipeKG


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def test_core_layer_tool_and_implementation():
    task_name = _uid("task")
    tool_name = _uid("tool")
    impl_name = _uid("impl")

    task_id = PipeKG.add_task(TaskEntity(name=task_name, description="test task"))
    tool_id = PipeKG.add_tool(ToolEntity(name=tool_name, supportsTasks=(task_id,)))
    implementation = ImplementationEntity(
        name=impl_name,
        version="1.0.0",
        input_spec=[],
        output_spec=[],
        realizesTask=[task_id],
        usesTool=[tool_id],
    )
    PipeKG.add_implementation(implementation)

    found = PipeKG.find_implementation(impl_name)
    assert len(found) == 1
    found_implementation = found[0]

    assert found_implementation.name == impl_name
    assert found_implementation.version == "1.0.0"
    assert task_id in found_implementation.realizesTask
    assert tool_id in found_implementation.usesTool


def test_resolve_data_spec_formats_returns_linked_data_type_format():
    data_type_id = PipeKG.add_data_type(
        DataTypeEntity(format="nt", data_schema="nt")
    )
    spec_id = PipeKG.add_data_spec(
        DataSpecEntity(name=_uid("input"), data_type=data_type_id)
    )

    assert PipeKG.resolve_data_spec_formats([spec_id]) == ["nt"]


def test_data_layer_type_spec_and_entity():
    schema_name = _uid("schema")
    spec_name = _uid("spec")
    artifact_uri = f"file:///{_uid('artifact')}.csv"

    data_type_id = PipeKG.add_data_type(
        DataTypeEntity(format="text/csv", data_schema=schema_name)
    )
    spec_id = PipeKG.add_data_spec(
        DataSpecEntity(name=spec_name, data_type=data_type_id)
    )
    data_id = PipeKG.add_data_entity(
        DataEntity(
            location=artifact_uri,
            data_type=data_type_id,
            version="1.0.0",
            hash="abc123",
            size=42,
        )
    )

    assert data_type_id
    assert spec_id
    assert data_id


def test_metrics_layer_add_metric():
    metric_name = _uid("metric")
    metric = MetricEntity(name=metric_name, description="Accuracy metric", type="score")
    # add_metric is not yet implemented; ensure the entity model is valid.
    assert metric.name == metric_name
    assert metric.description == "Accuracy metric"
    assert metric.type == "score"


def test_run_layer_add_task_run():
    task_name = _uid("task")
    impl_name = _uid("impl")
    artifact_uri_in = f"file:///{_uid('in')}.csv"
    artifact_uri_out = f"file:///{_uid('out')}.csv"

    task_id = PipeKG.add_task(TaskEntity(name=task_name))
    data_type_id = PipeKG.add_data_type(
        DataTypeEntity(format="text/csv", data_schema=_uid("schema"))
    )
    in_id = PipeKG.add_data_entity(
        DataEntity(location=artifact_uri_in, data_type=data_type_id)
    )
    out_id = PipeKG.add_data_entity(
        DataEntity(location=artifact_uri_out, data_type=data_type_id)
    )
    impl_id = PipeKG.add_implementation(
        ImplementationEntity(
            name=impl_name,
            version="1.0.0",
            input_spec=[],
            output_spec=[],
            realizesTask=[task_id],
            usesTool=[],
        )
    )

    task_run_id = PipeKG.add_task_run(
        TaskRunEntity(
            status="success",
            started_at=1.0,
            ended_at=2.0,
            input=[in_id],
            output=[out_id],
            usesImplementation=impl_id,
        )
    )

    assert task_run_id


def test_discovered_tasks_sync_to_systemgraph():
    discover_entry_points()

    paris_impl = PipeKG.find_implementation("paris_entity_matching")
    assert len(paris_impl) == 1
    assert paris_impl[0].name == "paris_entity_matching"
    assert len(paris_impl[0].input_spec) >= 1
    assert len(paris_impl[0].output_spec) >= 1

    registered_names = {task.name for task in get_registered_tasks()}
    synced_names = {impl.name for impl in PipeKG.find_implementation()}
    assert registered_names.issubset(synced_names)
