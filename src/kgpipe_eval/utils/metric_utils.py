from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

JsonValue = Any


@dataclass(frozen=True)
class MeasurementKey:
    metric: str
    measurement: str
    unit: str


Allowlist = Mapping[str, Mapping[str, str]]

from kgpipe_eval.api import MetricResult


def render_metric_result(metric_result: MetricResult, truncate: bool = False, truncate_value: int = 5) -> str:
    """
    Render a MetricResult into a human-readable table-like string.

    This is intended for CLI/test output (not machine-parseable export).
    """

    def _metric_key(mr: MetricResult) -> str:
        metric = mr.metric
        return getattr(metric, "key", metric.__class__.__name__)

    def _fmt_value(v: Any) -> str:
        if isinstance(v, float):
            # stable, compact representation for test output
            return f"{v:.6g}"
        if isinstance(v, (int, bool)) or v is None:
            return str(v)
        if isinstance(v, str):
            if truncate:
                lines = v.splitlines()[:truncate_value]
                return "\n".join(lines) + "\n..."
            return v
        if isinstance(v, Mapping):
            rendered = json.dumps(v, indent=2, sort_keys=True, default=str)
            if truncate:
                return "\n".join(rendered.splitlines()[:truncate_value]) + "\n..."
            return rendered
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            rendered = json.dumps(v, indent=2, sort_keys=True, default=str)
            if truncate:
                return "\n".join(rendered.splitlines()[:truncate_value]) + "\n..."
            return rendered
        return str(v)

    key = _metric_key(metric_result)
    summary = metric_result.summary or ""

    ms = sorted(metric_result.measurements, key=lambda m: m.name)
    name_w = max([len("measurement"), *(len(m.name) for m in ms)] or [len("measurement")])
    unit_w = max([len("unit"), *(len(m.unit or "") for m in ms)] or [len("unit")])

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"metric: {key}")
    if summary:
        lines.append(f"summary: {summary}")
    if not ms:
        lines.append("(no measurements)")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"{'measurement':<{name_w}}  {'value'}{' ' * max(1, 2)}{'unit':<{unit_w}}")
    lines.append(f"{'-' * name_w}  {'-' * 20}  {'-' * unit_w}")

    for m in ms:
        unit = m.unit or ""
        rendered = _fmt_value(m.value)
        rendered_lines = rendered.splitlines() or [""]
        lines.append(f"{m.name:<{name_w}}  {rendered_lines[0]:<20}  {unit:<{unit_w}}")
        for cont in rendered_lines[1:]:
            lines.append(f"{'':<{name_w}}  {cont}")

    return "\n".join(lines)


def parse_eval_results(path: Path) -> dict[MeasurementKey, JsonValue]:
    """
    Parse a single `eval_results.json` and return a flattened mapping.

    Expected file schema (per entry):
      - metric: str
      - measurements: [{name: str, value: any-json, unit: str|null}, ...]
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list, got {type(raw).__name__}")

    out: dict[MeasurementKey, JsonValue] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path} entries must be objects, got {type(entry).__name__}")

        metric = entry.get("metric")
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"{path} entry missing 'metric' string")

        measurements = entry.get("measurements", [])
        if not isinstance(measurements, list):
            raise ValueError(f"{path} entry 'measurements' must be a list")

        for m in measurements:
            if not isinstance(m, Mapping):
                continue
            name = m.get("name")
            unit = m.get("unit")
            if not isinstance(name, str) or not name:
                continue
            if unit is None:
                unit = ""
            if not isinstance(unit, str):
                unit = str(unit)
            out[MeasurementKey(metric=metric, measurement=name, unit=unit)] = m.get("value")

    return out


def allowlist_to_columns(allowlist: Allowlist) -> list[str]:
    cols: list[str] = []
    for metric in sorted(allowlist.keys()):
        for measurement in sorted(allowlist[metric].keys()):
            unit = allowlist[metric][measurement]
            cols.append(f"{metric}__{measurement}__{unit}")
    return cols


def eval_results_jsons_to_rows(
    paths: Sequence[Path],
    *,
    allowlist: Allowlist,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path in paths:
        if path.name != "eval_results.json":
            raise ValueError(f"Expected eval_results.json file, got {path}")
        stage_dir = path.parent
        stage = stage_dir.name
        if not stage.startswith("stage_"):
            raise ValueError(f"Expected stage directory named stage_*, got {stage_dir}")

        pipeline_dir = stage_dir.parent
        pipeline = pipeline_dir.name
        if not pipeline:
            raise ValueError(f"Could not derive pipeline name from {path}")

        flat = parse_eval_results(path)

        row: dict[str, Any] = {"pipeline": pipeline, "stage": stage}
        for metric, measurements in allowlist.items():
            for measurement, unit in measurements.items():
                key = MeasurementKey(metric=metric, measurement=measurement, unit=unit)
                col = f"{metric}__{measurement}__{unit}"
                row[col] = flat.get(key, "")

        rows.append(row)

    return rows


def write_eval_csv(
    paths: Sequence[Path],
    *,
    out_path: Path,
    allowlist: Allowlist,
    delimiter: str = ",",
    round_ndigits: int | None = None,
) -> None:
    rows = eval_results_jsons_to_rows(paths, allowlist=allowlist)
    columns = ["pipeline", "stage", *allowlist_to_columns(allowlist)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", delimiter=delimiter)
        writer.writeheader()
        for r in rows:
            # Ensure blanks for missing keys
            row = {k: r.get(k, "") for k in columns}
            if round_ndigits is not None:
                for k, v in list(row.items()):
                    if isinstance(v, float):
                        row[k] = round(v, round_ndigits)
            writer.writerow(row)

