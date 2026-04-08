import click
from rich.console import Console
from rich.table import Table
from typing import List, Optional, Sequence, Any
import json
from pathlib import Path

from kgpipe_eval.metrics.statistics import CountMetric
from kgpipe_eval.metrics.duplicates import DuplicateMetric
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric
from kgpipe_eval.utils.kg_utils import KgManager
from kgpipe_eval.config.manager import load_metric_configs
from kgpipe_eval.evaluator import Evaluator
# from kgpipe_eval.metrics.semantic import OntologyClassCoverageMetric, OntologyRelationCoverageMetric, OntologyNamespaceCoverageMetric
# from kgpipe_eval.metrics.reference import PrecisionMetric, RecallMetric, F1ScoreMetric
# from kgpipe_eval.metrics.efficiency import RuntimeMetric, MemoryUsageMetric, CostMetric
# from kgpipe_eval.metrics.quality import QualityMetric
# from kgpipe_eval.metrics.completeness import CompletenessMetric
# from kgpipe_eval.metrics.accuracy import AccuracyMetric

console = Console()


def _available_metric_instances() -> dict[str, Any]:
    # Keep this explicit until the metrics package is more complete/stable.
    return {
        "CountMetric": CountMetric(),
        "DuplicateMetric": DuplicateMetric(),
        "EntityAlignmentMetric": EntityAlignmentMetric(),
    }

def _normalize_key(k: str) -> str:
    return k.strip().lower().replace("-", "_")


def _metric_key(metric: Any) -> str:
    return getattr(metric, "key", metric.__class__.__name__)


def _build_confs_for_selected_metrics(
    selected_metric_instances: list[Any],
    loaded_confs: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert configs loaded from YAML (keyed by YAML metric id) into a dict keyed by
    metric class name / `.key` (what Evaluator uses).
    """
    confs_by_norm = {_normalize_key(k): v for k, v in loaded_confs.items()}
    out: dict[str, Any] = {}

    # Common YAML → class-name aliases
    alias_to_metric_key: dict[str, str] = {
        "duplicates": "DuplicateMetric",
        "duplicate": "DuplicateMetric",
        "entity_align": "EntityAlignmentMetric",
        "entity_alignment": "EntityAlignmentMetric",
    }

    for metric in selected_metric_instances:
        mkey = _metric_key(metric)
        norm_mkey = _normalize_key(mkey)
        norm_cls = _normalize_key(metric.__class__.__name__)

        cfg = (
            confs_by_norm.get(norm_mkey)
            or confs_by_norm.get(norm_cls)
            or confs_by_norm.get(_normalize_key(alias_to_metric_key.get(norm_mkey, "")))
            or confs_by_norm.get(_normalize_key(alias_to_metric_key.get(norm_cls, "")))
            or confs_by_norm.get(norm_mkey.replace("metric", ""))
            or confs_by_norm.get(norm_cls.replace("metric", ""))
        )

        if cfg is not None:
            out[mkey] = cfg
            out[metric.__class__.__name__] = cfg

    return out



def _render_results_table(kg_path: str, metric_key: str, measurements: Sequence[Any], summary: Optional[str]) -> None:
    table = Table(title=f"{Path(kg_path).name} — {metric_key}")
    table.add_column("Measurement", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="magenta")

    for m in measurements:
        unit = getattr(m, "unit", None)
        value = getattr(m, "value", None)
        name = getattr(m, "name", None)
        table.add_row(str(name), json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, (str, int, float, bool)) else str(value), "" if unit is None else str(unit))

    console.print(table)
    if summary:
        console.print(f"[dim]{summary}[/dim]")
    console.print("")


def _results_to_json_rows(kg_path: str, metric_key: str, measurements: Sequence[Any], summary: Optional[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in measurements:
        rows.append(
            {
                "kg_path": kg_path,
                "metric": metric_key,
                "measurement": getattr(m, "name", None),
                "value": getattr(m, "value", None),
                "unit": getattr(m, "unit", None),
                "summary": summary,
            }
        )
    return rows


@click.command()
@click.argument("kg_paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True), 
    help="Path to metric config file"
)
@click.option(
    "--metrics", 
    "-m", 
    multiple=True, 
    type=click.Choice(sorted(_available_metric_instances().keys())),
    help="Metrics to compute"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Write results to a JSON file (list of measurement rows).",
)
@click.pass_context
def eval_new_cmd(ctx: click.Context, kg_paths: List[str], config: Optional[str], metrics: tuple, output: Optional[str]):
    """
    Compute selected metrics for one or more KGs.

    KG_PATHS: one or more RDF files/directories that RDFLib can parse.
    """
    metric_instances = _available_metric_instances()
    selected_metrics = list(metrics) if metrics else list(metric_instances.keys())

    unknown = [m for m in selected_metrics if m not in metric_instances]
    if unknown:
        raise click.ClickException(f"Unknown metrics: {', '.join(unknown)}")

    loaded_metric_confs: dict[str, Any] = {}
    if config:
        loaded_metric_confs = load_metric_configs(config)

    all_rows: list[dict[str, Any]] = []

    for kg_path in kg_paths:
        console.print(f"[bold blue]Evaluating:[/bold blue] {kg_path}")
        kg_graph = KgManager.load_kg_from_path(Path(kg_path))
        try:
            selected_metric_instances = [metric_instances[k] for k in selected_metrics]
            confs = _build_confs_for_selected_metrics(selected_metric_instances, loaded_metric_confs)

            results = Evaluator().run(kg=kg_graph, metrics=selected_metric_instances, confs=confs)
            for res in results:
                metric_key = _metric_key(res.metric)
                _render_results_table(kg_path, metric_key, res.measurements, getattr(res, "summary", None))
                all_rows.extend(_results_to_json_rows(kg_path, metric_key, res.measurements, getattr(res, "summary", None)))
        finally:
            KgManager.unload_kg(kg_graph)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        console.print(f"[green]✓ Saved results to[/green] {output}")