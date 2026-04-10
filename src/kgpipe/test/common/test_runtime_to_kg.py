from pathlib import Path

from kgpipe.common.config import config
from kgpipe.common.models import Data, DataFormat, KgTaskReport
from kgpipe.common.model.task import KgTask
from kgpipe.common.runtime_to_kg import (
    data_to_handle,
    reports_to_pipeline_run_entity,
    task_to_task_entity,
    task_report_to_task_run_entity,
)


def _make_report(name: str, start_ts: float, duration: float, status: str = "success") -> KgTaskReport:
    return KgTaskReport(
        task_name=name,
        inputs=[Data(path=Path(f"{name}.in.nt"), format=DataFormat.RDF_NTRIPLES)],
        outputs=[Data(path=Path(f"{name}.out.nt"), format=DataFormat.RDF_NTRIPLES)],
        start_ts=start_ts,
        duration=duration,
        status=status,
    )


def test_data_to_handle_maps_path_and_format():
    data = Data(path=Path("test.nt"), format=DataFormat.RDF_NTRIPLES)
    handle = data_to_handle(data)
    assert handle.uri == "test.nt"
    assert handle.type == DataFormat.RDF_NTRIPLES


def test_task_to_task_entity_maps_name_and_defaults():
    task = KgTask(
        name="normalize",
        input_spec={"in": DataFormat.RDF_NTRIPLES},
        output_spec={"out": DataFormat.RDF_NTRIPLES},
        function=lambda _i, _o: None,
    )
    entity = task_to_task_entity(task)
    assert entity.name == "normalize"
    assert entity.hasSubtask == []


def test_task_report_to_task_run_entity_maps_core_fields():
    report = _make_report("normalize", start_ts=10.0, duration=2.5)
    entity = task_report_to_task_run_entity(report, index=3)

    assert entity.number == 3
    assert entity.name == "normalize"
    assert entity.status == "success"
    assert entity.started_at == 10.0
    assert entity.ended_at == 12.5
    assert str(entity.executesTask) == f"{config.PIPEKG_PREFIX}normalize"
    assert str(entity.usesImplementation) == f"{config.PIPEKG_PREFIX}normalizeImpl"
    assert len(entity.input) == 1
    assert len(entity.output) == 1


def test_reports_to_pipeline_run_entity_aggregates_times_and_runs():
    reports = [
        _make_report("step_a", start_ts=100.0, duration=10.0),
        _make_report("step_b", start_ts=80.0, duration=5.0),
    ]

    pipeline_entity = reports_to_pipeline_run_entity(reports, pipeline_name="demo_pipe")

    assert pipeline_entity.name == "demo_pipe"
    assert pipeline_entity.status == "success"
    assert pipeline_entity.started_at == 80.0
    assert pipeline_entity.ended_at == 110.0
    assert len(pipeline_entity.hasTaskRun) == 2
    assert pipeline_entity.hasTaskRun[0].number == 0
    assert pipeline_entity.hasTaskRun[1].number == 1


def test_reports_to_pipeline_run_entity_handles_empty_reports():
    pipeline_entity = reports_to_pipeline_run_entity([], pipeline_name="empty_pipe")

    assert pipeline_entity.name == "empty_pipe"
    assert pipeline_entity.status == "success"
    assert pipeline_entity.started_at == 0.0
    assert pipeline_entity.ended_at == 0.0
    assert pipeline_entity.hasTaskRun == []
