from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from kgpipe_eval.api import MetricResult
from kgpipe_eval.utils.metric_utils import MeasurementKey, parse_eval_results

JsonMapping = Mapping[str, Any]
MeasurementLookup = Mapping[MeasurementKey, Any]


@dataclass(frozen=True)
class ResolvedMeasurement:
    metric: str
    measurement: str
    value: float
    weight: float = 1.0
    transform: str | None = None


@dataclass(frozen=True)
class SubgroupScore:
    name: str
    score: float
    measurements: tuple[ResolvedMeasurement, ...] = ()


@dataclass(frozen=True)
class AggregateScore:
    final_score: float
    subgroups: dict[str, SubgroupScore] = field(default_factory=dict)


_AGGREGATIONS = frozenset(
    {"mean", "weighted_mean", "min", "max", "geometric_mean", "harmonic_mean", "product"}
)
_TRANSFORMS = frozenset({None, "identity", "invert", "one_minus"})


def _as_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: boolean values are not supported")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{context}: expected numeric value, got {type(value).__name__}")


def _apply_transform(value: float, transform: str | None) -> float:
    if transform in (None, "identity"):
        return value
    if transform in ("invert", "one_minus"):
        return 1.0 - value
    raise ValueError(f"Unsupported transform: {transform!r}")


def _aggregate(values: Sequence[float], method: str, weights: Sequence[float] | None = None) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty list of values")

    if method == "mean":
        return sum(values) / len(values)

    if method == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean requires weights")
        if len(weights) != len(values):
            raise ValueError("weighted_mean requires one weight per value")
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("weighted_mean requires positive total weight")
        return sum(v * w for v, w in zip(values, weights)) / total_weight

    if method == "min":
        return min(values)

    if method == "max":
        return max(values)

    if method == "product":
        result = 1.0
        for value in values:
            result *= value
        return result

    if method == "geometric_mean":
        if any(v < 0 for v in values):
            raise ValueError("geometric_mean requires non-negative values")
        if any(v == 0 for v in values):
            return 0.0
        return math.exp(sum(math.log(v) for v in values) / len(values))

    if method == "harmonic_mean":
        if any(v < 0 for v in values):
            raise ValueError("harmonic_mean requires non-negative values")
        if any(v == 0 for v in values):
            return 0.0
        return len(values) / sum(1.0 / v for v in values)

    raise ValueError(f"Unsupported aggregation method: {method!r}")


def _parse_measurement_ref(item: Any, *, subgroup: str) -> tuple[str, str, float, str | None]:
    if isinstance(item, str):
        if "." in item:
            metric, measurement = item.split(".", 1)
        elif ":" in item:
            metric, measurement = item.split(":", 1)
        else:
            raise ValueError(
                f"subgroup {subgroup!r}: measurement ref {item!r} must be "
                "'MetricName.measurement' or 'MetricName:measurement'"
            )
        return metric, measurement, 1.0, None

    if not isinstance(item, Mapping):
        raise ValueError(f"subgroup {subgroup!r}: measurement item must be a mapping or string")

    metric = item.get("metric")
    measurement = item.get("measurement")
    if not isinstance(metric, str) or not metric:
        raise ValueError(f"subgroup {subgroup!r}: measurement item missing 'metric'")
    if not isinstance(measurement, str) or not measurement:
        raise ValueError(f"subgroup {subgroup!r}: measurement item missing 'measurement'")

    weight = item.get("weight", 1.0)
    transform = item.get("transform")
    if not isinstance(weight, (int, float)):
        raise ValueError(f"subgroup {subgroup!r}: weight for {metric}.{measurement} must be numeric")
    if transform is not None and not isinstance(transform, str):
        raise ValueError(f"subgroup {subgroup!r}: transform for {metric}.{measurement} must be a string")
    if transform not in _TRANSFORMS:
        raise ValueError(f"subgroup {subgroup!r}: unsupported transform {transform!r}")

    return metric, measurement, float(weight), transform


def _lookup_measurement(
    measurements: MeasurementLookup,
    *,
    metric: str,
    measurement: str,
    subgroup: str,
) -> Any:
    for key, value in measurements.items():
        if key.metric == metric and key.measurement == measurement:
            return value
    raise KeyError(
        f"subgroup {subgroup!r}: measurement {metric}.{measurement} not found in eval results"
    )


def _resolve_subgroup(
    name: str,
    subgroup_cfg: JsonMapping,
    measurements: MeasurementLookup,
) -> SubgroupScore:
    if not isinstance(subgroup_cfg, Mapping):
        raise ValueError(f"subgroup {name!r}: config must be an object")

    items = subgroup_cfg.get("measurements", subgroup_cfg.get("items", []))
    if not isinstance(items, list) or not items:
        raise ValueError(f"subgroup {name!r}: 'measurements' must be a non-empty list")

    aggregation = subgroup_cfg.get("aggregation", subgroup_cfg.get("type", "mean"))
    if not isinstance(aggregation, str) or aggregation not in _AGGREGATIONS:
        raise ValueError(f"subgroup {name!r}: unsupported aggregation {aggregation!r}")

    default_transform = subgroup_cfg.get("transform")
    if default_transform is not None and default_transform not in _TRANSFORMS:
        raise ValueError(f"subgroup {name!r}: unsupported transform {default_transform!r}")

    resolved: list[ResolvedMeasurement] = []
    values: list[float] = []
    weights: list[float] = []

    for item in items:
        metric, measurement, weight, item_transform = _parse_measurement_ref(item, subgroup=name)
        transform = item_transform if item_transform is not None else default_transform
        raw_value = _lookup_measurement(
            measurements,
            metric=metric,
            measurement=measurement,
            subgroup=name,
        )
        value = _apply_transform(
            _as_float(raw_value, context=f"{name}.{metric}.{measurement}"),
            transform,
        )
        resolved.append(
            ResolvedMeasurement(
                metric=metric,
                measurement=measurement,
                value=value,
                weight=weight,
                transform=transform,
            )
        )
        values.append(value)
        weights.append(weight)

    score = _aggregate(values, aggregation, weights if aggregation == "weighted_mean" else None)
    return SubgroupScore(name=name, score=score, measurements=tuple(resolved))


def aggregate_scores(
    measurements: MeasurementLookup | Sequence[Mapping[str, Any]] | Path | str,
    config: JsonMapping,
) -> AggregateScore:
    """
    Aggregate eval measurements into named subgroups, then into a final score.

    Config schema (dict / JSON):

    {
      "subgroups": {
        "coverage": {
          "measurements": [
            {"metric": "EntityAlignmentMetric", "measurement": "recall"},
            {"metric": "TripleAlignmentMetric", "measurement": "recall", "weight": 2.0}
          ],
          "aggregation": "mean"
        },
        "correctness": {
          "measurements": [
            "EntityAlignmentMetric.precision",
            "TripleAlignmentMetric.precision"
          ],
          "aggregation": "mean"
        },
        "cleanliness": {
          "measurements": [
            {"metric": "DuplicateMetric", "measurement": "duplicates_ratio", "transform": "invert"}
          ],
          "aggregation": "mean"
        }
      },
      "final": {
        "aggregation": "weighted_mean",
        "weights": {
          "coverage": 0.4,
          "correctness": 0.4,
          "cleanliness": 0.2
        }
      }
    }

    Measurement refs may be objects or shorthand strings like ``MetricName.measurement``.
    Supported subgroup/final aggregations: mean, weighted_mean, min, max, geometric_mean,
    harmonic_mean, product.
    Supported transforms: identity (default), invert / one_minus (``1 - value``).
    """
    lookup = _coerce_measurement_lookup(measurements)

    subgroups_cfg = config.get("subgroups")
    if not isinstance(subgroups_cfg, Mapping) or not subgroups_cfg:
        raise ValueError("config must contain a non-empty 'subgroups' object")

    subgroup_scores: dict[str, SubgroupScore] = {}
    for name, subgroup_cfg in subgroups_cfg.items():
        if not isinstance(name, str) or not name:
            raise ValueError("subgroup names must be non-empty strings")
        subgroup_scores[name] = _resolve_subgroup(name, subgroup_cfg, lookup)

    final_cfg = config.get("final", {})
    if not isinstance(final_cfg, Mapping):
        raise ValueError("config 'final' must be an object")

    final_aggregation = final_cfg.get("aggregation", final_cfg.get("type", "weighted_mean"))
    if not isinstance(final_aggregation, str) or final_aggregation not in _AGGREGATIONS:
        raise ValueError(f"unsupported final aggregation {final_aggregation!r}")

    subgroup_names = list(subgroup_scores.keys())
    subgroup_values = [subgroup_scores[name].score for name in subgroup_names]

    final_weights_cfg = final_cfg.get("weights")
    if final_aggregation == "weighted_mean":
        if not isinstance(final_weights_cfg, Mapping):
            raise ValueError("final weighted_mean requires a 'weights' object")
        final_weights = [float(final_weights_cfg.get(name, 0.0)) for name in subgroup_names]
    elif isinstance(final_weights_cfg, Mapping):
        final_weights = [float(final_weights_cfg.get(name, 1.0)) for name in subgroup_names]
    else:
        final_weights = None

    final_score = _aggregate(
        subgroup_values,
        final_aggregation,
        final_weights if final_aggregation == "weighted_mean" else None,
    )
    return AggregateScore(final_score=final_score, subgroups=subgroup_scores)


def aggregate_scores_from_json(
    eval_results_path: Path | str,
    config: JsonMapping | Path | str,
) -> AggregateScore:
    """Load eval results and config from JSON files and compute the aggregate score."""
    measurements = parse_eval_results(Path(eval_results_path))
    resolved_config = _coerce_config(config)
    return aggregate_scores(measurements, resolved_config)

def aggregate_scores_from_results(
    results: List[MetricResult],
    config: JsonMapping | Path | str,
) -> AggregateScore:
    lookup: dict[MeasurementKey, Any] = {}
    for result in results:
        metric = getattr(result.metric, "key", result.metric.__class__.__name__)
        for measurement in result.measurements:
            lookup[
                MeasurementKey(
                    metric=metric,
                    measurement=measurement.name,
                    unit=measurement.unit or "",
                )
            ] = measurement.value
    resolved_config = _coerce_config(config)
    return aggregate_scores(lookup, resolved_config)

def _coerce_measurement_lookup(
    measurements: MeasurementLookup | Sequence[Mapping[str, Any]] | Path | str,
) -> dict[MeasurementKey, Any]:
    if isinstance(measurements, (str, Path)):
        return parse_eval_results(Path(measurements))

    if isinstance(measurements, Sequence) and not isinstance(measurements, (str, bytes, bytearray)):
        if measurements and isinstance(measurements[0], Mapping) and "metric" in measurements[0]:
            out: dict[MeasurementKey, Any] = {}
            for entry in measurements:
                if not isinstance(entry, Mapping):
                    continue
                metric = entry.get("metric")
                if not isinstance(metric, str):
                    continue
                for m in entry.get("measurements", []):
                    if not isinstance(m, Mapping):
                        continue
                    name = m.get("name")
                    unit = m.get("unit") or ""
                    if isinstance(name, str) and name:
                        out[MeasurementKey(metric=metric, measurement=name, unit=str(unit))] = m.get("value")
            return out
        return dict(measurements)  # type: ignore[arg-type]

    return dict(measurements)


def _coerce_config(config: JsonMapping | Path | str) -> JsonMapping:
    if isinstance(config, (str, Path)):
        path = Path(config)
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        return loaded
    return config
