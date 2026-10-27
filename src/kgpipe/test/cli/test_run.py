from pathlib import Path
from uuid import uuid4

from click.testing import CliRunner

from kgpipe.cli.run import resolve_pipeline_names, run_cmd
from kgpipe.common.graph.systemgraph import PipeKG
from kgpipe.common.model.task import KgTask
from kgpipe.common.models import DataFormat
from kgpipe.common.registry import Registry


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def test_run_dry_run_registers_pipeline(monkeypatch, tmp_path: Path):
    task_a = KgTask(
        name=_uid("cli_task_a"),
        input_spec={"source": DataFormat.JSON, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"matches": DataFormat.ER_JSON},
        function=lambda _i, _o: None,
    )
    task_b = KgTask(
        name=_uid("cli_task_b"),
        input_spec={"matches": DataFormat.ER_JSON, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"result": DataFormat.RDF_NTRIPLES},
        function=lambda _i, _o: None,
    )
    Registry.add_task(task_a.name, task_a)
    Registry.add_task(task_b.name, task_b)

    pipeline_name = _uid("pipeline")
    pipeline_conf = tmp_path / "pipeline.conf"
    pipeline_conf.write_text(
        "\n".join(
            [
                f"{pipeline_name}:",
                '    description: "test pipeline"',
                "    tasks:",
                f"        - {task_a.name}",
                f"        - {task_b.name}",
                "",
            ]
        )
    )

    monkeypatch.setattr("kgpipe.cli.run.discover_entry_points", lambda: None)

    runner = CliRunner()
    result = runner.invoke(
        run_cmd,
        [str(pipeline_conf), pipeline_name, "--dry-run", "--data-dir", str(tmp_path / "work")],
    )

    assert result.exit_code == 0, result.output
    assert "Registered pipeline" in result.output
    assert len(PipeKG.find_pipeline(pipeline_name)) == 1


def _write_two_pipeline_conf(tmp_path: Path, task_a: KgTask, task_b: KgTask) -> tuple[Path, str, str]:
    pipeline_a = _uid("pipeline_a")
    pipeline_b = _uid("pipeline_b")
    pipeline_conf = tmp_path / "pipeline.conf"
    pipeline_conf.write_text(
        "\n".join(
            [
                f"{pipeline_a}:",
                '    description: "pipeline a"',
                "    tasks:",
                f"        - {task_a.name}",
                "",
                f"{pipeline_b}:",
                '    description: "pipeline b"',
                "    tasks:",
                f"        - {task_b.name}",
                "",
            ]
        )
    )
    return pipeline_conf, pipeline_a, pipeline_b


def test_run_dry_run_multiple_positional(monkeypatch, tmp_path: Path):
    task_a = KgTask(
        name=_uid("cli_task_a"),
        input_spec={"source": DataFormat.JSON, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"matches": DataFormat.ER_JSON},
        function=lambda _i, _o: None,
    )
    task_b = KgTask(
        name=_uid("cli_task_b"),
        input_spec={"source": DataFormat.TEXT, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"result": DataFormat.RDF_NTRIPLES},
        function=lambda _i, _o: None,
    )
    Registry.add_task(task_a.name, task_a)
    Registry.add_task(task_b.name, task_b)

    pipeline_conf, pipeline_a, pipeline_b = _write_two_pipeline_conf(tmp_path, task_a, task_b)
    monkeypatch.setattr("kgpipe.cli.run.discover_entry_points", lambda: None)

    runner = CliRunner()
    result = runner.invoke(
        run_cmd,
        [
            str(pipeline_conf),
            pipeline_a,
            pipeline_b,
            "--dry-run",
            "--data-dir",
            str(tmp_path / "work"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dry Run Summary" in result.output
    assert len(PipeKG.find_pipeline(pipeline_a)) == 1
    assert len(PipeKG.find_pipeline(pipeline_b)) == 1


def test_run_dry_run_all(monkeypatch, tmp_path: Path):
    task_a = KgTask(
        name=_uid("cli_task_a"),
        input_spec={"source": DataFormat.JSON, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"matches": DataFormat.ER_JSON},
        function=lambda _i, _o: None,
    )
    task_b = KgTask(
        name=_uid("cli_task_b"),
        input_spec={"source": DataFormat.TEXT, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"result": DataFormat.RDF_NTRIPLES},
        function=lambda _i, _o: None,
    )
    Registry.add_task(task_a.name, task_a)
    Registry.add_task(task_b.name, task_b)

    pipeline_conf, pipeline_a, pipeline_b = _write_two_pipeline_conf(tmp_path, task_a, task_b)
    monkeypatch.setattr("kgpipe.cli.run.discover_entry_points", lambda: None)

    runner = CliRunner()
    result = runner.invoke(
        run_cmd,
        [str(pipeline_conf), "--all", "--dry-run", "--data-dir", str(tmp_path / "work")],
    )

    assert result.exit_code == 0, result.output
    assert "Registered 2 pipeline(s)" in result.output
    assert len(PipeKG.find_pipeline(pipeline_a)) == 1
    assert len(PipeKG.find_pipeline(pipeline_b)) == 1


def test_run_dry_run_accepts_input_target_and_ontology(monkeypatch, tmp_path: Path):
    task = KgTask(
        name=_uid("cli_task"),
        input_spec={"source": DataFormat.JSON, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"result": DataFormat.RDF_NTRIPLES},
        function=lambda _i, _o: None,
    )
    Registry.add_task(task.name, task)

    pipeline_name = _uid("pipeline")
    pipeline_conf = tmp_path / "pipeline.conf"
    pipeline_conf.write_text(
        "\n".join(
            [
                f"{pipeline_name}:",
                '    description: "with io"',
                "    tasks:",
                f"        - {task.name}",
                "",
            ]
        )
    )
    source = tmp_path / "source.json"
    seed = tmp_path / "seed.nt"
    ontology = tmp_path / "ontology.ttl"
    source.write_text("{}\n")
    seed.write_text("\n")
    ontology.write_text("@prefix ex: <http://example.org/> .\n")
    output_dir = tmp_path / "output"

    monkeypatch.setattr("kgpipe.cli.run.discover_entry_points", lambda: None)

    runner = CliRunner()
    result = runner.invoke(
        run_cmd,
        [
            str(pipeline_conf),
            "--pipeline",
            pipeline_name,
            "--input",
            str(source),
            "--target",
            str(seed),
            "--ontology",
            str(ontology),
            "--output",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Input: {source}" in result.output
    assert f"Target: {seed}" in result.output
    assert f"Ontology: {ontology}" in result.output
    assert f"Output: {output_dir}" in result.output
    assert (output_dir / pipeline_name).is_dir()

    from kgpipe.cli.run import build_dry_run_pipeline

    pipeline = build_dry_run_pipeline(
        pipeline_file=str(pipeline_conf),
        pipeline_name=pipeline_name,
        data_dir=str(output_dir),
        source_file=str(source),
        target_file=str(seed),
        ontology_file=str(ontology),
    )
    assert pipeline.plan.source is not None
    assert pipeline.plan.seed is not None
    assert pipeline.plan.source.path == source
    assert pipeline.plan.seed.path == seed
    assert pipeline.plan.source.format == DataFormat.JSON
    assert pipeline.plan.seed.format == DataFormat.RDF_NTRIPLES
    assert pipeline.ontology_path == str(ontology)


def test_resolve_pipeline_names_all_excludes_explicit_names(tmp_path: Path):
    pipeline_conf = tmp_path / "pipeline.conf"
    pipeline_conf.write_text("alpha:\n    description: a\n    tasks: []\n")

    try:
        resolve_pipeline_names(str(pipeline_conf), ("alpha",), (), True)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "either --all or explicit" in str(exc).lower()
