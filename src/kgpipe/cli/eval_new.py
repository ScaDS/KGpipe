import click
from rich.console import Console
from rich.table import Table
from typing import List, Optional, Sequence, Any
import json
from pathlib import Path
import codecs

from kgpipe_eval.metrics.statistics import CountMetric
from kgpipe_eval.metrics.duplicates import DuplicateMetric
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric
from kgpipe_eval.utils.kg_utils import KgManager
from kgpipe_eval.utils.metric_utils import MeasurementKey, parse_eval_results, write_eval_csv
from kgpipe_eval.config.manager import load_metric_configs, write_default_config_yaml
from kgpipe_eval.evaluator import Evaluator
# from kgpipe_eval.metrics.semantic import OntologyClassCoverageMetric, OntologyRelationCoverageMetric, OntologyNamespaceCoverageMetric
# from kgpipe_eval.metrics.reference import PrecisionMetric, RecallMetric, F1ScoreMetric
# from kgpipe_eval.metrics.efficiency import RuntimeMetric, MemoryUsageMetric, CostMetric
# from kgpipe_eval.metrics.quality import QualityMetric
# from kgpipe_eval.metrics.completeness import CompletenessMetric
# from kgpipe_eval.metrics.accuracy import AccuracyMetric

console = Console()

_DEFAULT_EVAL_RESULTS_ALLOWLIST = {
    "DuplicateMetric": {
        "duplicates": "number",
        "entity_count": "number",
        "duplicates_ratio": "percentage",
    }
}

def _measurement_key_to_col(k: MeasurementKey) -> str:
    return f"{k.metric}__{k.measurement}__{k.unit}"


def _col_to_measurement_key(col: str) -> MeasurementKey:
    parts = col.split("__")
    if len(parts) != 3 or not all(parts):
        raise click.ClickException(
            f"Invalid selection '{col}'. Expected format: <metric>__<measurement>__<unit>"
        )
    return MeasurementKey(metric=parts[0], measurement=parts[1], unit=parts[2])


def _available_eval_result_keys(paths: list[Path]) -> list[MeasurementKey]:
    keys: set[MeasurementKey] = set()
    for p in paths:
        flat = parse_eval_results(p)
        keys.update(flat.keys())
    return sorted(keys, key=_measurement_key_to_col)

def _decode_single_char_delimiter(delimiter: str) -> str:
    """
    Allow passing common escape sequences like '\\t' for tab.
    """
    decoded = codecs.decode(delimiter, "unicode_escape") if "\\" in delimiter else delimiter
    if len(decoded) != 1:
        raise click.ClickException(
            f"--delimiter must be a single character (you passed {delimiter!r} -> {decoded!r})"
        )
    return decoded


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

        # Try common YAML ids derived from metric names
        base_from_key = norm_mkey.replace("_metric", "").replace("metric", "")
        base_from_cls = norm_cls.replace("_metric", "").replace("metric", "")

        cfg = (
            confs_by_norm.get(norm_mkey)
            or confs_by_norm.get(norm_cls)
            or confs_by_norm.get(_normalize_key(alias_to_metric_key.get(norm_mkey, "")))
            or confs_by_norm.get(_normalize_key(alias_to_metric_key.get(norm_cls, "")))
            or confs_by_norm.get(base_from_key)
            or confs_by_norm.get(base_from_cls)
            # plural fallback (e.g. DuplicateMetric -> duplicates)
            or confs_by_norm.get(f"{base_from_key}s")
            or confs_by_norm.get(f"{base_from_cls}s")
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


@click.group(name="eval-new")
def eval_new_cmd() -> None:
    """
    Evaluation commands for the new metric framework.
    """


@eval_new_cmd.command(name="run")
@click.argument("kg_paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to metric config file",
)
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    type=click.Choice(sorted(_available_metric_instances().keys())),
    help="Metrics to compute",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Write results to a JSON file (list of measurement rows).",
)
@click.pass_context
def run_cmd(ctx: click.Context, kg_paths: List[str], config: Optional[str], metrics: tuple, output: Optional[str]) -> None:
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


@eval_new_cmd.command(name="init-config")
@click.argument("output_path", type=click.Path(dir_okay=False), default="eval.default.yaml", required=False)
def init_config_cmd(output_path: str) -> None:
    """
    Write a default metric-config template YAML to OUTPUT_PATH.
    """
    out = write_default_config_yaml(output_path)
    console.print(f"[green]✓ Wrote default config to[/green] {out}")


@eval_new_cmd.command(name="to-csv")
@click.argument("eval_json_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--glob",
    "glob_pattern",
    type=str,
    help="Optional glob pattern (expanded by the shell) for eval_results.json files.",
)
@click.option(
    "--select",
    "-s",
    "selected_cols",
    multiple=True,
    help="Select columns to include (repeatable). Format: <metric>__<measurement>__<unit>. If omitted, defaults are used.",
)
@click.option(
    "--list-keys",
    is_flag=True,
    help="Print available column keys found in the inputs and exit.",
)
@click.option(
    "--round",
    "round_ndigits",
    type=int,
    default=None,
    help="Round float values to N decimal digits before writing CSV.",
)
@click.option(
    "--delimiter",
    "delimiter",
    type=str,
    default=",",
    show_default=True,
    help="CSV delimiter character (supports escapes like '\\t').",
)
@click.option(
    "--output",
    "-o",
    "output_csv",
    type=click.Path(dir_okay=False),
    required=True,
    help="Path to write the CSV table to.",
)
def to_csv_cmd(
    eval_json_paths: List[str],
    glob_pattern: Optional[str],
    selected_cols: tuple[str, ...],
    list_keys: bool,
    round_ndigits: Optional[int],
    delimiter: str,
    output_csv: str,
) -> None:
    """
    Convert one or more `eval_results.json` files into a CSV table.

    The CSV contains one row per (pipeline, stage), derived from file paths like:
      `<pipeline>/stage_<i>/eval_results.json`

    Columns follow: `<metric>__<measurement>__<unit>`.
    """
    paths: list[Path] = [Path(p) for p in eval_json_paths]
    if glob_pattern:
        paths.extend(sorted(Path().glob(glob_pattern)))

    if not paths:
        raise click.ClickException("No input files provided. Pass paths or --glob.")

    available = _available_eval_result_keys(paths)
    console.print("[bold]Available keys in inputs:[/bold]")
    for k in available:
        console.print(f"  - {_measurement_key_to_col(k)}")

    if list_keys:
        return

    allowlist = _DEFAULT_EVAL_RESULTS_ALLOWLIST
    if selected_cols:
        available_cols = {_measurement_key_to_col(k) for k in available}
        missing = [c for c in selected_cols if c not in available_cols]
        if missing:
            raise click.ClickException(
                "Selected keys not found in inputs:\n" + "\n".join(f"- {m}" for m in missing)
            )

        allowlist = {}
        for c in selected_cols:
            k = _col_to_measurement_key(c)
            allowlist.setdefault(k.metric, {})[k.measurement] = k.unit

    out_path = Path(output_csv)
    delimiter = _decode_single_char_delimiter(delimiter)
    write_eval_csv(
        paths,
        out_path=out_path,
        allowlist=allowlist,
        delimiter=delimiter,
        round_ndigits=round_ndigits,
    )
    console.print(f"[green]✓ Wrote CSV to[/green] {out_path}")