import inspect
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import yaml
from click.testing import CliRunner

from kgpipe.cli.exec import exec_cmd, plan_cmd, validate_cmd
from kgpipe.cli.main import cli
from kgpipe.common.models import DataFormat, KgTask
from kgpipe.common.registry import Registry

EXAMPLES = Path(__file__).resolve().parents[4] / "experiments" / "examples"
DUMMY_TASKS = EXAMPLES / "src" / "kgpipe_examples" / "dummy_tasks.py"
SAMPLE = EXAMPLES / "sample.yaml"
CATALOG = EXAMPLES / "pipeline.conf"


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def _invoke(args: list[str], **kwargs):
    return CliRunner().invoke(cli, args, **kwargs)


def _json(result) -> dict:
    assert result.stdout.strip().startswith("{"), result.stdout
    return json.loads(result.stdout)


def test_validate_json_in_a_clean_process():
    """Import-time diagnostics must not prefix stdout. CliRunner starts too late to see them."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kgpipe.cli.main",
            "validate",
            str(SAMPLE),
            "--module-path",
            str(DUMMY_TASKS),
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["name"] == "sample"
    assert payload["status"] == "valid"
    assert "Adding task" not in result.stdout
    assert "validate:" in result.stderr


def test_command_options_match_callbacks():
    for command in (validate_cmd, plan_cmd, exec_cmd):
        callback = inspect.unwrap(command.callback)
        names = set(inspect.signature(callback).parameters) - {"ctx"}
        assert {param.name for param in command.params} == names


def test_help_documents_validate_plan_and_exec():
    runner = CliRunner()
    for command in ("validate", "plan", "exec"):
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--format" in result.stdout
        assert "--module-path" in result.stdout
        assert "--mode" not in result.stdout
    dry_run = runner.invoke(cli, ["exec", "--help"])
    assert "--dry-run" in dry_run.stdout
    assert "--force" in dry_run.stdout


def test_example_validate_json_is_stdout_only():
    result = _invoke(
        [
            "validate",
            str(SAMPLE),
            "--module-path",
            str(DUMMY_TASKS),
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.stderr
    payload = _json(result)
    assert payload["command"] == "validate"
    assert payload["name"] == "sample"
    assert payload["status"] == "valid"
    assert payload["fingerprint"]
    assert payload["steps"][0]["parameters"]["entity_matching_threshold"] == 0.8
    assert payload["steps"][2]["parameters"]["entity_matching_threshold"] == 0.99
    assert "validate:" in result.stderr
    assert "validate:" not in result.stdout


def test_example_plan_prints_auto_wiring():
    result = _invoke(
        [
            "plan",
            str(CATALOG),
            "dummy_auto",
            "--module-path",
            str(DUMMY_TASKS),
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.stderr
    payload = _json(result)
    assert [(step["inputs"], step["outputs"]) for step in payload["steps"]] == [
        ({"source": "$source", "kg": "$seed"}, {"output": "dummy_match.output"}),
        ({"input": "dummy_match.output"}, {"output": "dummy_exchange.output"}),
        (
            {"source": "$source", "kg": "$seed", "matches1": "dummy_exchange.output"},
            {"output": "dummy_fuse.output"},
        ),
        ({"source": "dummy_fuse.output"}, {"result": "$result"}),
    ]


def test_example_catalog_requires_a_name():
    result = _invoke(
        [
            "validate",
            str(CATALOG),
            "--module-path",
            str(DUMMY_TASKS),
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 2, result.stdout
    payload = _json(result)
    assert payload["status"] == "invalid"
    assert "pipeline name" in payload["error"]
    assert payload["error"] in result.stderr


def test_example_exec_dry_run_does_not_run_tasks(tmp_path: Path):
    result_file = tmp_path / "result.nt"
    data_dir = tmp_path / "work"
    result = _invoke(
        [
            "exec",
            str(SAMPLE),
            "--module-path",
            str(DUMMY_TASKS),
            "--input",
            str(tmp_path / "missing-source.nt"),
            "--seed",
            str(tmp_path / "missing-seed.nt"),
            "--result",
            str(result_file),
            "--data-dir",
            str(data_dir),
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.stderr
    payload = _json(result)
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert "reports" not in payload
    assert not result_file.exists()
    assert not data_dir.exists()


def test_example_exec_runs_and_second_run_skips(tmp_path: Path):
    source = tmp_path / "source.nt"
    seed = tmp_path / "seed.nt"
    source.write_text("<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n")
    seed.write_text("<http://example.org/a> <http://example.org/p> <http://example.org/b> .\n")
    result_file = tmp_path / "out" / "result.nt"
    args = [
        "exec",
        str(SAMPLE),
        "--module-path",
        str(DUMMY_TASKS),
        "--input",
        str(source),
        "--seed",
        str(seed),
        "--result",
        str(result_file),
        "--data-dir",
        str(tmp_path / "out"),
        "--format",
        "json",
    ]
    first = _invoke(args)
    assert first.exit_code == 0, first.stderr
    payload = _json(first)
    assert payload["status"] == "success"
    assert payload["reports"]
    assert all(report["status"] == "success" for report in payload["reports"])
    assert result_file.exists()
    assert "Running " not in first.stdout
    assert "Running " in first.stderr

    second = _invoke(args)
    assert second.exit_code == 0, second.stderr
    skipped = _json(second)
    assert skipped["status"] == "skipped"
    assert all(report["status"] == "skipped" for report in skipped["reports"])

    forced = _invoke(args + ["--force"])
    assert forced.exit_code == 0, forced.stderr
    assert _json(forced)["status"] == "success"


def test_example_exec_missing_input_exits_1(tmp_path: Path):
    result = _invoke(
        [
            "exec",
            str(SAMPLE),
            "--module-path",
            str(DUMMY_TASKS),
            "--input",
            str(tmp_path / "missing.nt"),
            "--seed",
            str(tmp_path / "seed.nt"),
            "--result",
            str(tmp_path / "result.nt"),
            "--data-dir",
            str(tmp_path / "work"),
            "--format",
            "yaml",
        ]
    )
    assert result.exit_code == 1, result.stderr
    payload = yaml.safe_load(result.stdout)
    assert payload["status"] == "failed"
    assert "does not exist" in payload["error"]
    assert result.stdout.strip().startswith("command:")
    assert "Running " not in result.stdout


def test_quiet_json_has_empty_stderr(tmp_path: Path):
    source = tmp_path / "source.nt"
    seed = tmp_path / "seed.nt"
    source.touch()
    seed.touch()
    result_file = tmp_path / "result.nt"
    result = _invoke(
        [
            "--quiet",
            "exec",
            str(SAMPLE),
            "--module-path",
            str(DUMMY_TASKS),
            "--input",
            str(source),
            "--seed",
            str(seed),
            "--result",
            str(result_file),
            "--data-dir",
            str(tmp_path / "work"),
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert _json(result)["status"] == "success"
    assert result_file.exists()


def test_unknown_task_exits_2_and_text_stays_off_stdout(tmp_path: Path):
    catalog = tmp_path / "pipeline.yaml"
    catalog.write_text(
        "\n".join(
            [
                "schema: kgpipe.pipeline/v1",
                "broken:",
                '  description: "missing task"',
                "  tasks:",
                "    - not_a_registered_task",
                "",
            ]
        )
    )
    result = _invoke(["validate", str(catalog), "--format", "text"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Unknown task" in result.stderr

    machine = _invoke(["validate", str(catalog), "--format", "json"])
    assert machine.exit_code == 2
    payload = _json(machine)
    assert payload["status"] == "invalid"
    assert "Unknown task" in payload["error"]
    assert "Unknown task" in machine.stderr


def test_failing_task_exits_1(tmp_path: Path):
    name = _uid("boom")

    def _boom(_inputs, _outputs):
        raise RuntimeError("nope")

    Registry.add_task(
        name,
        KgTask(
            name=name,
            input_spec={"source": DataFormat.RDF_NTRIPLES},
            output_spec={"result": DataFormat.RDF_NTRIPLES},
            function=_boom,
        ),
    )
    catalog = tmp_path / "pipeline.yaml"
    catalog.write_text(
        "\n".join(
            [
                "schema: kgpipe.pipeline/v1",
                "defaults:",
                f"  source: {tmp_path / 'source.nt'}",
                f"  seed: {tmp_path / 'seed.nt'}",
                f"  result: {tmp_path / 'result.nt'}",
                f"  data_dir: {tmp_path / 'work'}",
                "run:",
                '  description: "fails"',
                "  tasks:",
                f"    - {name}",
                "",
            ]
        )
    )
    (tmp_path / "source.nt").touch()
    (tmp_path / "seed.nt").touch()
    result = _invoke(["exec", str(catalog), "--format", "json"])
    assert result.exit_code == 1, result.stderr
    payload = _json(result)
    assert payload["status"] == "failed"
    assert "nope" in (payload["error"] or "")
    assert "Running " not in result.stdout
    assert "An error occurred" not in result.stdout
