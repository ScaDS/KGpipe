#!/usr/bin/env python3
"""
Run command for KGpipe CLI.

This module currently supports pipeline dry-runs that materialize a
`KgPipePlan` from a pipeline catalog entry and sync its abstract structure
into the system graph.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from kgpipe.common.discovery import discover_entry_points
from kgpipe.common.graph.mapper import sync_pipeline_to_systemgraph
from kgpipe.common.models import Data, DataFormat, KgPipe
from kgpipe.generation.loaders import build_from_conf, load_pipeline_catalog

# Initialize Rich console for pretty output
console = Console()


def show_pipeline_plan(pipeline: KgPipe):
    """Display pipeline execution plan."""
    table = Table(title="Pipeline Execution Plan")
    table.add_column("Task", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Input", style="green")
    table.add_column("Output", style="yellow")
    
    for i, task in enumerate(pipeline.tasks, 1):
        table.add_row(
            f"{i}. {task.name}",
            task.__class__.__name__,
            str(task.input_spec) if hasattr(task, 'input_spec') else "N/A",
            str(task.output_spec) if hasattr(task, 'output_spec') else "N/A"
        )
    
    console.print(table)


def _infer_source_format(pipeline: KgPipe) -> DataFormat:
    first_task = pipeline.tasks[0]
    for input_name, format_spec in first_task.input_spec.items():
        if input_name != "kg":
            return format_spec
    return DataFormat.ANY


def _infer_seed_format(pipeline: KgPipe) -> DataFormat:
    first_task = pipeline.tasks[0]
    if "kg" in first_task.input_spec:
        return first_task.input_spec["kg"]
    return pipeline.seed.format


def _infer_result_format(pipeline: KgPipe) -> DataFormat:
    last_task = pipeline.tasks[-1]
    output_formats = list(last_task.output_spec.values())
    return output_formats[0] if output_formats else DataFormat.ANY


def _format_for_path(path: str | None, fallback: DataFormat) -> DataFormat:
    """Use a recognized file extension, otherwise the format inferred from the pipeline."""
    if not path:
        return fallback
    suffix = Path(path).suffix.lower().lstrip(".")
    if not suffix:
        return fallback
    try:
        return DataFormat.from_extension(suffix)
    except ValueError:
        return fallback


def resolve_pipeline_names(
    pipeline_file: str,
    pipeline_names: tuple[str, ...],
    pipelines: tuple[str, ...],
    run_all: bool,
) -> list[str]:
    catalog = load_pipeline_catalog(Path(pipeline_file))
    available = sorted(catalog.root.keys())

    if run_all:
        if pipeline_names or pipelines:
            raise ValueError("Use either --all or explicit pipeline names, not both.")
        return available

    selected = [*pipeline_names, *pipelines]
    if not selected:
        raise ValueError(
            "No pipeline selected. Pass PIPELINE_NAME, use --pipeline/-p, or --all.\n"
            f"Available: {', '.join(available)}"
        )

    unknown = sorted({name for name in selected if name not in catalog.root})
    if unknown:
        raise ValueError(
            f"Unknown pipeline(s): {', '.join(unknown)}. Available: {', '.join(available)}"
        )

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(selected))


def build_dry_run_pipeline(
    pipeline_file: str,
    pipeline_name: str,
    data_dir: str | None = None,
    source_file: str | None = None,
    target_file: str | None = None,
    ontology_file: str | None = None,
) -> KgPipe:
    catalog = load_pipeline_catalog(Path(pipeline_file))
    if pipeline_name not in catalog.root:
        available = ", ".join(sorted(catalog.root.keys()))
        raise ValueError(f"Unknown pipeline '{pipeline_name}'. Available: {available}")

    pipeline_conf = catalog.root[pipeline_name]
    base_dir = Path(data_dir) if data_dir else Path(".kgpipe") / "dry-run"
    working_dir = base_dir / pipeline_name
    working_dir.mkdir(parents=True, exist_ok=True)

    seed_path = Path(target_file) if target_file else working_dir / "seed.placeholder"
    seed_data = Data(
        path=seed_path,
        format=_format_for_path(target_file, DataFormat.RDF_NTRIPLES),
    )
    pipeline = build_from_conf(
        name=pipeline_name,
        conf=pipeline_conf,
        target_data=seed_data,
        data_dir=working_dir.as_posix(),
    )

    source_path = Path(source_file) if source_file else working_dir / "source.placeholder"
    source_data = Data(
        path=source_path,
        format=_format_for_path(source_file, _infer_source_format(pipeline)),
    )
    pipeline.seed = Data(
        path=seed_path,
        format=_format_for_path(target_file, _infer_seed_format(pipeline)),
    )
    result_data = Data(
        path=working_dir / "result.placeholder",
        format=_infer_result_format(pipeline),
    )
    pipeline.data = [pipeline.seed]
    if ontology_file:
        pipeline.ontology_path = ontology_file
    pipeline.build(source=source_data, result=result_data, stable_files=True)
    return pipeline


def dry_run_pipelines(
    pipeline_file: str,
    pipeline_names: list[str],
    data_dir: str | None = None,
    source_file: str | None = None,
    target_file: str | None = None,
    ontology_file: str | None = None,
) -> list[tuple[str, str | None, str | None]]:
    """Dry-run pipelines and return (name, pipeline_id, error) tuples."""
    results: list[tuple[str, str | None, str | None]] = []
    for pipeline_name in pipeline_names:
        try:
            pipeline = build_dry_run_pipeline(
                pipeline_file=pipeline_file,
                pipeline_name=pipeline_name,
                data_dir=data_dir,
                source_file=source_file,
                target_file=target_file,
                ontology_file=ontology_file,
            )
            pipeline_id = sync_pipeline_to_systemgraph(pipeline)
            results.append((pipeline_name, str(pipeline_id), None))
        except Exception as exc:
            results.append((pipeline_name, None, str(exc)))
    return results


def _print_run_inputs(
    source_file: str | None,
    target_file: str | None,
    ontology_file: str | None,
    data_dir: str | None,
) -> None:
    if source_file:
        console.print(f"[dim]Input:[/dim] {source_file}", soft_wrap=True)
    if target_file:
        console.print(f"[dim]Target:[/dim] {target_file}", soft_wrap=True)
    if ontology_file:
        console.print(f"[dim]Ontology:[/dim] {ontology_file}", soft_wrap=True)
    if data_dir:
        console.print(f"[dim]Output:[/dim] {data_dir}", soft_wrap=True)


def show_dry_run_summary(results: list[tuple[str, str | None, str | None]]) -> None:
    table = Table(title="Dry Run Summary")
    table.add_column("Pipeline", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Pipeline entity", style="yellow")

    for pipeline_name, pipeline_id, error in results:
        if error is None:
            table.add_row(pipeline_name, "registered", pipeline_id or "—")
        else:
            table.add_row(pipeline_name, f"failed: {error}", "—")

    console.print(table)


@click.command(name="run")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.argument("pipeline_names", nargs=-1)
@click.option(
    "--pipeline",
    "-p",
    "pipelines",
    multiple=True,
    help="Pipeline name(s) from the catalog. Can be repeated.",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    help="Dry-run all pipelines defined in the catalog file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Build the pipeline plan and register it in the system graph without execution.",
)
@click.option(
    "--input",
    "source_file",
    type=click.Path(),
    help="Source file consumed by the pipeline (SOURCE_FILE).",
)
@click.option(
    "--target",
    "target_file",
    type=click.Path(),
    help="Seed knowledge graph (SEED_KG).",
)
@click.option(
    "--ontology",
    "ontology_file",
    type=click.Path(),
    help="Optional ontology file (ONTOLOGY).",
)
@click.option(
    "--output",
    "--data-dir",
    "data_dir",
    type=click.Path(),
    help="Directory for planned pipeline outputs.",
)
@click.pass_context
def run_cmd(
    ctx: click.Context,
    pipeline_file: str,
    pipeline_names: tuple[str, ...],
    pipelines: tuple[str, ...],
    run_all: bool,
    dry_run: bool,
    source_file: str | None,
    target_file: str | None,
    ontology_file: str | None,
    data_dir: str | None,
):
    """
    Build or execute pipeline catalog entries.

    PIPELINE_FILE: Path to a `pipeline.conf`-style catalog file
    PIPELINE_NAMES: Optional one or more pipeline names from that catalog

    Examples:

        kgpipe run pipeline.conf rdf_base --dry-run
        kgpipe run pipeline.conf --pipeline rdf_base --input source.nt --target seed.nt --output ./output --dry-run
        kgpipe run pipeline.conf --pipeline rdf_base --input source.nt --target seed.nt --ontology ontology.ttl --output ./output --dry-run
        kgpipe run pipeline.conf --all --dry-run
    """
    if not dry_run:
        console.print("[red]Only `--dry-run` is supported right now.[/red]")
        console.print(
            "Use `kgpipe run PIPELINE_FILE [PIPELINE_NAME...] --dry-run` "
            "or `kgpipe run PIPELINE_FILE --all --dry-run`."
        )
        sys.exit(1)

    try:
        selected = resolve_pipeline_names(
            pipeline_file=pipeline_file,
            pipeline_names=pipeline_names,
            pipelines=pipelines,
            run_all=run_all,
        )
        discover_entry_points()
        _print_run_inputs(source_file, target_file, ontology_file, data_dir)
        if len(selected) == 1:
            pipeline_name = selected[0]
            pipeline = build_dry_run_pipeline(
                pipeline_file=pipeline_file,
                pipeline_name=pipeline_name,
                data_dir=data_dir,
                source_file=source_file,
                target_file=target_file,
                ontology_file=ontology_file,
            )
            pipeline_id = sync_pipeline_to_systemgraph(pipeline)
            console.print("[yellow]DRY RUN - Showing execution plan:[/yellow]")
            show_pipeline_plan(pipeline)
            console.print(f"[green]✓[/green] Registered pipeline `{pipeline_name}` in the system graph.")
            console.print(f"[dim]Pipeline entity:[/dim] {pipeline_id}")
            return

        results = dry_run_pipelines(
            pipeline_file=pipeline_file,
            pipeline_names=selected,
            data_dir=data_dir,
            source_file=source_file,
            target_file=target_file,
            ontology_file=ontology_file,
        )
        console.print(
            f"[yellow]DRY RUN - Registering {len(selected)} pipeline(s) in the system graph[/yellow]"
        )
        show_dry_run_summary(results)
        failed = [name for name, pipeline_id, error in results if error is not None]
        if failed:
            console.print(f"[red]✗[/red] Failed pipelines: {', '.join(failed)}")
            sys.exit(1)
        console.print(f"[green]✓[/green] Registered {len(selected)} pipeline(s) in the system graph.")
    except Exception as e:
        console.print(f"[red]✗ Pipeline dry-run failed:[/red] {e}")
        if ctx.obj["verbose"]:
            console.print_exception()
        sys.exit(1)