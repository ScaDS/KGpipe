from pathlib import Path

from kgpipe.common.models import Data, DataFormat, KgTask


def _write_output_task(inputs: dict[str, Data], outputs: dict[str, Data]) -> None:
    _ = inputs["in"]
    out_path = outputs["out"].path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("generated")


def test_kgtask_run_success(tmp_path: Path):
    in_file = tmp_path / "input.nt"
    out_file = tmp_path / "output.nt"
    in_file.write_text("seed")

    task = KgTask(
        name="copy_like",
        input_spec={"in": DataFormat.RDF_NTRIPLES},
        output_spec={"out": DataFormat.RDF_NTRIPLES},
        function=_write_output_task,
    )

    report = task.run(
        inputs=[Data(path=in_file, format=DataFormat.RDF_NTRIPLES)],
        outputs=[Data(path=out_file, format=DataFormat.RDF_NTRIPLES)],
    )

    assert report.status == "success"
    assert out_file.exists()
    assert report.task_name == "copy_like"
    assert len(report.inputs) == 1
    assert len(report.outputs) == 1


def test_kgtask_run_failed_when_function_raises(tmp_path: Path):
    def failing_task(_: dict[str, Data], __: dict[str, Data]) -> None:
        raise RuntimeError("boom")

    in_file = tmp_path / "input.nt"
    out_file = tmp_path / "output.nt"
    in_file.write_text("seed")

    task = KgTask(
        name="fails",
        input_spec={"in": DataFormat.RDF_NTRIPLES},
        output_spec={"out": DataFormat.RDF_NTRIPLES},
        function=failing_task,
    )

    report = task.run(
        inputs=[Data(path=in_file, format=DataFormat.RDF_NTRIPLES)],
        outputs=[Data(path=out_file, format=DataFormat.RDF_NTRIPLES)],
    )

    assert report.status == "failed"
    assert report.error is not None
    assert "boom" in report.error


def test_kgtask_run_skips_when_outputs_exist(tmp_path: Path):
    called = {"count": 0}

    def should_not_run(_: dict[str, Data], __: dict[str, Data]) -> None:
        called["count"] += 1

    in_file = tmp_path / "input.nt"
    out_file = tmp_path / "output.nt"
    in_file.write_text("seed")
    out_file.write_text("already-here")

    task = KgTask(
        name="skip_if_present",
        input_spec={"in": DataFormat.RDF_NTRIPLES},
        output_spec={"out": DataFormat.RDF_NTRIPLES},
        function=should_not_run,
    )

    report = task.run(
        inputs=[Data(path=in_file, format=DataFormat.RDF_NTRIPLES)],
        outputs=[Data(path=out_file, format=DataFormat.RDF_NTRIPLES)],
    )

    assert report.status == "skipped"
    assert called["count"] == 0


def test_kgtask_stable_files_override_forces_run(tmp_path: Path):
    called = {"count": 0}
    out_file = tmp_path / "output.nt"

    def rewrite_output(_: dict[str, Data], outputs: dict[str, Data]) -> None:
        called["count"] += 1
        outputs["out"].path.write_text("fresh")

    in_file = tmp_path / "input.nt"
    in_file.write_text("seed")
    out_file.write_text("stale")

    task = KgTask(
        name="override_output",
        input_spec={"in": DataFormat.RDF_NTRIPLES},
        output_spec={"out": DataFormat.RDF_NTRIPLES},
        function=rewrite_output,
    )

    report = task.run(
        inputs=[Data(path=in_file, format=DataFormat.RDF_NTRIPLES)],
        outputs=[Data(path=out_file, format=DataFormat.RDF_NTRIPLES)],
        stable_files_override=True,
    )

    assert report.status == "success"
    assert called["count"] == 1
    assert out_file.read_text() == "fresh"


def test_kgtask_run_fails_for_missing_required_input(tmp_path: Path):
    out_file = tmp_path / "output.nt"

    task = KgTask(
        name="needs_input",
        input_spec={"in": DataFormat.RDF_NTRIPLES},
        output_spec={"out": DataFormat.RDF_NTRIPLES},
        function=_write_output_task,
    )

    report = task.run(
        inputs=[],
        outputs=[Data(path=out_file, format=DataFormat.RDF_NTRIPLES)],
    )

    assert report.status == "failed"
    assert report.error is not None
    assert "Missing required inputs" in report.error
