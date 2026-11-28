from uuid import uuid4
from pathlib import Path

from kgpipe.common.discovery import discover_entry_points, get_registered_tasks, get_registered_metrics
from kgpipe.common.graph.definitions import (
    DataEntity,
    KGPIPE_NS,
    DataSpecEntity,
    DataTypeEntity,
    ImplementationEntity,
    MeasurementSpecEntity,
    MetricEntity,
    TaskEntity,
    TaskRunEntity,
    ToolEntity,
)
from kgpipe.common.graph.systemgraph import PipeKG, SYS_KG
from kgpipe.common.graph.mapper import sync_metric_to_systemgraph, sync_pipeline_to_systemgraph
from kgpipe.common.model.pipeline import KgPipe
from kgpipe.common.model.task import KgTask
from kgpipe.common.models import Data, DataFormat
from kgpipe_eval.metrics.statistics import CountMetric


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


def test_resolve_data_spec_ports_preserves_same_format_multiplicity():
    data_type_id = PipeKG.add_data_type(
        DataTypeEntity(format="te.json", data_schema="te.json")
    )
    spec_ids = [
        PipeKG.add_data_spec(DataSpecEntity(name="json1", data_type=data_type_id)),
        PipeKG.add_data_spec(DataSpecEntity(name="json2", data_type=data_type_id)),
        PipeKG.add_data_spec(DataSpecEntity(name="json3", data_type=data_type_id)),
    ]

    ports = PipeKG.resolve_data_spec_ports(spec_ids)
    assert [(p["name"], p["format"]) for p in ports] == [
        ("json1", "te.json"),
        ("json2", "te.json"),
        ("json3", "te.json"),
    ]
    # Unique-format helper still collapses.
    assert PipeKG.resolve_data_spec_formats(spec_ids) == ["te.json"]


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
    m1 = PipeKG.add_measurement_spec(
        MeasurementSpecEntity(
            name="precision",
            unit="percentage",
            alias=("prec", "p"),
        )
    )
    m2 = PipeKG.add_measurement_spec(
        MeasurementSpecEntity(name="recall", unit="percentage")
    )
    metric_id = PipeKG.add_metric(
        MetricEntity(
            name=metric_name,
            description="Accuracy metric",
            type="score",
            measurements=[m1, m2],
        )
    )

    found = PipeKG.find_metric(metric_name)
    assert len(found) == 1
    assert found[0].name == metric_name
    assert found[0].description == "Accuracy metric"
    assert found[0].type == "score"
    assert len(found[0].measurements) == 2

    specs = PipeKG.resolve_measurement_specs(found[0].measurements)
    by_name = {s.name: s for s in specs}
    assert by_name["precision"].unit == "percentage"
    assert by_name["precision"].alias == ("prec", "p")
    assert by_name["recall"].unit == "percentage"
    assert by_name["recall"].alias == ()
    assert metric_id


def test_sync_metric_to_systemgraph_writes_measurement_specs():
    metric_id = sync_metric_to_systemgraph(CountMetric)
    found = PipeKG.find_metric(CountMetric.key)
    assert len(found) == 1
    assert found[0].name == CountMetric.key
    assert found[0].description == CountMetric.description

    specs = PipeKG.resolve_measurement_specs(found[0].measurements)
    expected = {m.name: m.unit for m in CountMetric.measurements}
    assert {s.name: s.unit for s in specs} == expected
    assert metric_id


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


def test_config_spec_roundtrip_on_implementation():
    from kgpipe.common.graph.mapper import implementation_to_entity
    from kgpipe.common.model.configuration import (
        ConfigurationDefinition,
        Parameter,
        ParameterType,
    )
    from kgpipe.common.model.task import KgTask
    from kgpipe.common.models import DataFormat

    task_name = _uid("cfg_task")
    config_name = _uid("cfg_spec")

    def _fn(inputs, outputs):
        return None

    task = KgTask(
        name=task_name,
        input_spec={"input": DataFormat.ANY},
        output_spec={"output": DataFormat.ANY},
        function=_fn,
        description="config roundtrip",
        config_spec=ConfigurationDefinition(
            name=config_name,
            description="demo config",
            parameters=[
                Parameter(
                    name="mode",
                    native_keys=["--mode"],
                    datatype=ParameterType.enum,
                    default_value="exact",
                    required=True,
                    allowed_values=["exact", "fuzzy"],
                ),
                Parameter(
                    name="threshold",
                    native_keys=["--threshold"],
                    datatype=ParameterType.number,
                    default_value=0.5,
                    required=False,
                    minimum=0.0,
                    maximum=1.0,
                    unit="ratio",
                ),
            ],
        ),
    )

    impl_id = implementation_to_entity(task)
    found = PipeKG.find_implementation(task_name)
    assert len(found) == 1
    assert found[0].config_spec is not None

    spec, params = PipeKG.resolve_config_spec_parameters(found[0].config_spec)
    assert spec is not None
    assert spec.name == config_name
    assert spec.description == "demo config"
    assert impl_id
    by_key = {p.key: p for p in params}
    assert by_key["mode"].datatype == "enum"
    assert by_key["mode"].allowed_values == ("exact", "fuzzy")
    assert by_key["threshold"].datatype == "number"
    assert by_key["threshold"].minimum == 0.0
    assert by_key["threshold"].maximum == 1.0
    assert by_key["threshold"].unit == "ratio"


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


def test_discovered_metrics_sync_to_systemgraph():
    discover_entry_points()

    count_metric = PipeKG.find_metric("CountMetric")
    assert len(count_metric) == 1
    assert count_metric[0].name == "CountMetric"
    specs = PipeKG.resolve_measurement_specs(count_metric[0].measurements)
    assert {s.name for s in specs} == {m.name for m in CountMetric.measurements}

    registered_keys = {getattr(m, "key", m.__name__) for m in get_registered_metrics()}
    synced_keys = {metric.name for metric in PipeKG.find_metric()}
    assert registered_keys.issubset(synced_keys)


def test_sync_pipeline_to_systemgraph_registers_abstract_steps():
    task_a = KgTask(
        name=_uid("pipe_task_a"),
        input_spec={"source": DataFormat.JSON, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"matches": DataFormat.ER_JSON},
        function=lambda _i, _o: None,
    )
    task_b = KgTask(
        name=_uid("pipe_task_b"),
        input_spec={"matches": DataFormat.ER_JSON, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"result": DataFormat.RDF_NTRIPLES},
        function=lambda _i, _o: None,
    )
    pipeline = KgPipe(
        tasks=[task_a, task_b],
        seed=Data(path=Path("seed.nt"), format=DataFormat.RDF_NTRIPLES),
        data_dir="/tmp/kgpipe-pipeline-test",
        name=_uid("pipeline"),
    )
    pipeline.build(
        source=Data(path=Path("source.json"), format=DataFormat.JSON),
        result=Data(path=Path("result.nt"), format=DataFormat.RDF_NTRIPLES),
        stable_files=True,
    )

    pipeline_id = sync_pipeline_to_systemgraph(pipeline)
    found = PipeKG.find_pipeline(pipeline.name)

    assert pipeline_id
    assert len(found) == 1
    assert found[0].name == pipeline.name
    assert len(found[0].steps) == 2

    step_entities = [SYS_KG.read_entity(str(step_id)) for step_id in found[0].steps]
    step_names = [entity.get_property_value(str(KGPIPE_NS.name))[0] for entity in step_entities if entity is not None]
    assert step_names == [f"1. {task_a.name}", f"2. {task_b.name}"]
