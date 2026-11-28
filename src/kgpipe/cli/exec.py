"""Validate, plan, and execute a pipeline catalog entry.

``validate``, ``plan``, and ``exec`` call :class:`kgpipe.Pipeline`. stdout is
the result. Progress from tasks and one-line status messages go to stderr.
``--format json`` and ``--format yaml`` never share stdout with that progress.

Exit codes:

- ``0`` — the catalog is valid, the plan was built, or execution succeeded
  or was skipped because outputs already existed
- ``1`` — a task failed, or execution stopped
- ``2`` — the catalog, task names, parameters, or ports are invalid
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import click

from kgpipe import Pipeline
from kgpipe.api import PipelineRun
from kgpipe.common.model.pipeline import KgPipePlan
from kgpipe.generation.spec import CatalogValidationError

EXIT_OK = 0
EXIT_EXECUTION = 1
EXIT_VALIDATION = 2


def _ctx_flag(ctx: click.Context, name: str) -> bool:
    obj = ctx.obj if isinstance(ctx.obj, dict) else {}
    return bool(obj.get(name))


def _note(ctx: click.Context, message: str, *, error: bool = False) -> None:
    """Write a diagnostic to stderr. ``--quiet`` hides success notes."""
    if not error and _ctx_flag(ctx, "quiet"):
        return
    click.echo(message, err=True)


@contextmanager
def _isolate_stdout(quiet: bool) -> Iterator[None]:
    """Keep task ``print`` calls off the result stream.

    Quiet mode discards them. Otherwise they go to stderr, next to the
    status line, and stdout stays free for text, JSON, or YAML.
    """
    saved = sys.stdout
    sink = open(os.devnull, "w") if quiet else None
    sys.stdout = sink if sink is not None else sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved
        if sink is not None:
            sink.close()


def _import_user_code(
    packages: tuple[str, ...],
    module_paths: tuple[Path, ...],
    discover: bool,
) -> None:
    for package in packages:
        try:
            importlib.import_module(package)
        except Exception as exc:
            raise CatalogValidationError(f"Could not import {package}: {exc}") from exc
    if module_paths:
        from kgpipe.common.discovery import discover_local_modules

        for module_path in module_paths:
            discover_local_modules(Path(module_path))
    if discover:
        from kgpipe.common.discovery import discover_entry_points

        discover_entry_points()


def _format_value(fmt: Any) -> str:
    return str(getattr(fmt, "value", fmt))


def _data(value: Any) -> Optional[dict[str, str]]:
    if value is None:
        return None
    return {"path": str(value.path), "format": _format_value(value.format)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _steps(plan: KgPipePlan) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for step in plan.steps:
        steps.append(
            {
                "id": step.id,
                "task": step.task,
                "inputs": dict(step.inputs),
                "outputs": dict(step.outputs),
                "parameters": _jsonable(dict(step.parameters)),
                "files": {
                    "inputs": [_data(item) for item in step.input],
                    "outputs": [_data(item) for item in step.output],
                },
            }
        )
    return steps


def _plan_payload(
    pipeline: Pipeline,
    *,
    command: str,
    status: str,
    error: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = pipeline.plan()
    payload: dict[str, Any] = {
        "command": command,
        "name": pipeline.name,
        "status": status,
        "fingerprint": plan.fingerprint,
        "connect": plan.connect,
        "error": error,
        "source": _data(plan.source),
        "seed": _data(plan.seed),
        "result": _data(plan.result),
        "steps": _steps(plan),
    }
    if dry_run:
        payload["dry_run"] = True
    return payload


def _run_payload(pipeline: Pipeline, outcome: PipelineRun) -> dict[str, Any]:
    payload = _plan_payload(
        pipeline,
        command="exec",
        status=outcome.status,
        error=outcome.error,
    )
    payload["duration"] = round(outcome.duration, 6)
    payload["result"] = _data(outcome.result)
    payload["reports"] = [
        {
            "task": report.task_name,
            "status": report.status,
            "duration": round(report.duration, 6),
            "error": report.error,
            "inputs": [_data(item) for item in report.inputs],
            "outputs": [_data(item) for item in report.outputs],
        }
        for report in outcome.reports
    ]
    return payload


def _error_payload(command: str, name: Optional[str], status: str, error: str) -> dict[str, Any]:
    message = error.replace(
        "Pass name= to select one.",
        "Pass the pipeline name after the file.",
    )
    return {
        "command": command,
        "name": name,
        "status": status,
        "error": message,
    }


def _format_text(payload: dict[str, Any]) -> str:
    name = payload.get("name") or ""
    lines = [f"{payload['status']}  {name}".rstrip()]
    if payload.get("fingerprint"):
        lines.append(f"fingerprint: {payload['fingerprint']}")
    if payload.get("duration") is not None:
        lines.append(f"duration: {payload['duration']:.3f}s")
    if payload.get("error"):
        lines.append(f"error: {payload['error']}")
    result = payload.get("result")
    if isinstance(result, dict) and result.get("path"):
        lines.append(f"result: {result['path']}")
    if payload.get("command") != "validate":
        for step in payload.get("steps") or []:
            step_id = step.get("id") or step.get("task")
            lines.append(f"{step_id}  {step.get('task')}")
            inputs = " ".join(f"{key}={value}" for key, value in (step.get("inputs") or {}).items())
            outputs = " ".join(f"{key}={value}" for key, value in (step.get("outputs") or {}).items())
            if inputs:
                lines.append(f"  in   {inputs}")
            if outputs:
                lines.append(f"  out  {outputs}")
            if step.get("parameters"):
                params = " ".join(f"{key}={value}" for key, value in step["parameters"].items())
                lines.append(f"  parameters  {params}")
    for report in payload.get("reports") or []:
        suffix = f"  {report['error']}" if report.get("error") else ""
        lines.append(f"{report['task']}  {report['status']}  {report['duration']:.3f}s{suffix}")
    return "\n".join(lines)


def _emit(payload: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(payload, sort_keys=False).rstrip())
        return
    click.echo(_format_text(payload))


def _load(
    pipeline_file: Path,
    name: Optional[str],
    *,
    source: Optional[Path],
    seed: Optional[Path],
    result_file: Optional[Path],
    data_dir: Optional[Path],
    ontology: Optional[Path],
    packages: tuple[str, ...],
    module_paths: tuple[Path, ...],
    discover: bool,
) -> Pipeline:
    _import_user_code(packages, module_paths, discover)
    return Pipeline.from_yaml(
        pipeline_file,
        name,
        source=source,
        seed=seed,
        result=result_file,
        data_dir=data_dir,
        ontology=ontology,
    )


def _execute(
    *,
    command: str,
    pipeline_file: Path,
    name: Optional[str],
    source: Optional[Path],
    seed: Optional[Path],
    result_file: Optional[Path],
    data_dir: Optional[Path],
    ontology: Optional[Path],
    packages: tuple[str, ...],
    module_paths: tuple[Path, ...],
    discover: bool,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[dict[str, Any], int]:
    pipeline = _load(
        pipeline_file,
        name,
        source=source,
        seed=seed,
        result_file=result_file,
        data_dir=data_dir,
        ontology=ontology,
        packages=packages,
        module_paths=module_paths,
        discover=discover,
    )
    if command == "exec" and not dry_run:
        outcome = pipeline.run(stable_files_override=force)
        code = EXIT_OK if outcome.status in {"success", "skipped"} else EXIT_EXECUTION
        return _run_payload(pipeline, outcome), code
    status = "valid" if command == "validate" else "planned"
    return (
        _plan_payload(pipeline, command=command, status=status, dry_run=dry_run),
        EXIT_OK,
    )


def _pipeline_options(command: click.Command | Any) -> Any:
    command = click.option(
        "--format",
        "output_format",
        type=click.Choice(["text", "json", "yaml"]),
        default="text",
        show_default=True,
        help="Result format on stdout. json and yaml are the document only.",
    )(command)
    command = click.option(
        "--discover",
        is_flag=True,
        help="Import installed KGpipe task packages before loading the catalog.",
    )(command)
    command = click.option(
        "--module-path",
        "module_paths",
        multiple=True,
        type=click.Path(exists=True, path_type=Path),
        help="Import tasks from this file or directory. Repeatable.",
    )(command)
    command = click.option(
        "--package",
        "packages",
        multiple=True,
        help="Import this module so its tasks register. Repeatable.",
    )(command)
    command = click.option(
        "--ontology",
        type=click.Path(path_type=Path),
        default=None,
        help="Ontology file. Overrides the catalog ontology.",
    )(command)
    command = click.option(
        "--output",
        "--data-dir",
        "data_dir",
        type=click.Path(path_type=Path),
        default=None,
        help="Directory for intermediate files. Overrides the catalog data_dir.",
    )(command)
    command = click.option(
        "--result",
        "result_file",
        type=click.Path(path_type=Path),
        default=None,
        help="File written by the last step. Overrides the catalog result.",
    )(command)
    command = click.option(
        "--seed",
        "--target",
        "seed",
        type=click.Path(path_type=Path),
        default=None,
        help="Seed knowledge graph. --target is the same file.",
    )(command)
    command = click.option(
        "--input",
        "--source",
        "source",
        type=click.Path(path_type=Path),
        default=None,
        help="Source file. Overrides the catalog source.",
    )(command)
    command = click.argument("name", required=False)(command)
    command = click.argument(
        "pipeline_file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )(command)
    return command


def _dispatch(
    ctx: click.Context,
    *,
    command: str,
    pipeline_file: Path,
    name: Optional[str],
    source: Optional[Path],
    seed: Optional[Path],
    result_file: Optional[Path],
    data_dir: Optional[Path],
    ontology: Optional[Path],
    packages: tuple[str, ...],
    module_paths: tuple[Path, ...],
    discover: bool,
    output_format: str,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    shown_name = name or pipeline_file.stem
    try:
        with _isolate_stdout(_ctx_flag(ctx, "quiet")):
            payload, code = _execute(
                command=command,
                pipeline_file=pipeline_file,
                name=name,
                source=source,
                seed=seed,
                result_file=result_file,
                data_dir=data_dir,
                ontology=ontology,
                packages=packages,
                module_paths=module_paths,
                discover=discover,
                dry_run=dry_run,
                force=force,
            )
    except CatalogValidationError as exc:
        payload = _error_payload(command, name, "invalid", str(exc))
        code = EXIT_VALIDATION
    except Exception as exc:
        if _ctx_flag(ctx, "verbose"):
            traceback.print_exc(file=sys.stderr)
        payload = _error_payload(command, name, "failed", f"{type(exc).__name__}: {exc}")
        code = EXIT_EXECUTION

    label = payload.get("name") or shown_name
    detail = payload.get("error") or payload["status"]
    _note(ctx, f"{command}: {label} {detail}", error=code != EXIT_OK)
    if output_format == "text" and code != EXIT_OK:
        ctx.exit(code)
    _emit(payload, output_format)
    if code != EXIT_OK:
        ctx.exit(code)


@click.command("validate")
@_pipeline_options
@click.pass_context
def validate_cmd(
    ctx: click.Context,
    pipeline_file: Path,
    name: Optional[str],
    source: Optional[Path],
    seed: Optional[Path],
    result_file: Optional[Path],
    data_dir: Optional[Path],
    ontology: Optional[Path],
    packages: tuple[str, ...],
    module_paths: tuple[Path, ...],
    discover: bool,
    output_format: str,
) -> None:
    """Check that a catalog entry resolves to a plan. Does not run tasks.

    PIPELINE_FILE is a kgpipe.pipeline/v1 catalog. NAME is required when the
    file defines more than one pipeline.

    Exit status is 0 when the plan resolves and 2 when it does not.
    """
    _dispatch(
        ctx,
        command="validate",
        pipeline_file=pipeline_file,
        name=name,
        source=source,
        seed=seed,
        result_file=result_file,
        data_dir=data_dir,
        ontology=ontology,
        packages=packages,
        module_paths=module_paths,
        discover=discover,
        output_format=output_format,
    )


@click.command("plan")
@_pipeline_options
@click.pass_context
def plan_cmd(
    ctx: click.Context,
    pipeline_file: Path,
    name: Optional[str],
    source: Optional[Path],
    seed: Optional[Path],
    result_file: Optional[Path],
    data_dir: Optional[Path],
    ontology: Optional[Path],
    packages: tuple[str, ...],
    module_paths: tuple[Path, ...],
    discover: bool,
    output_format: str,
) -> None:
    """Print the resolved plan: port refs, parameters, and fingerprint.

    PIPELINE_FILE is a kgpipe.pipeline/v1 catalog. NAME is required when the
    file defines more than one pipeline.

    Exit status is 0 when the plan resolves and 2 when it does not.
    """
    _dispatch(
        ctx,
        command="plan",
        pipeline_file=pipeline_file,
        name=name,
        source=source,
        seed=seed,
        result_file=result_file,
        data_dir=data_dir,
        ontology=ontology,
        packages=packages,
        module_paths=module_paths,
        discover=discover,
        output_format=output_format,
    )


@click.command("exec")
@_pipeline_options
@click.option(
    "--dry-run",
    is_flag=True,
    help="Resolve the plan and print it without executing tasks.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Run tasks even when their output files already exist.",
)
@click.pass_context
def exec_cmd(
    ctx: click.Context,
    pipeline_file: Path,
    name: Optional[str],
    source: Optional[Path],
    seed: Optional[Path],
    result_file: Optional[Path],
    data_dir: Optional[Path],
    ontology: Optional[Path],
    packages: tuple[str, ...],
    module_paths: tuple[Path, ...],
    discover: bool,
    output_format: str,
    dry_run: bool,
    force: bool,
) -> None:
    """Execute a catalog entry and print the run result.

    PIPELINE_FILE is a kgpipe.pipeline/v1 catalog. NAME is required when the
    file defines more than one pipeline. ``--dry-run`` stops after planning.

    Exit status is 0 on success or when every task was skipped, 1 when a task
    fails, and 2 when the catalog cannot be planned.
    """
    _dispatch(
        ctx,
        command="exec",
        pipeline_file=pipeline_file,
        name=name,
        source=source,
        seed=seed,
        result_file=result_file,
        data_dir=data_dir,
        ontology=ontology,
        packages=packages,
        module_paths=module_paths,
        discover=discover,
        output_format=output_format,
        dry_run=dry_run,
        force=force,
    )
