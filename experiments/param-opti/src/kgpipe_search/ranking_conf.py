"""Named score-aggregation configs used when ranking pipeline evaluation results."""

from __future__ import annotations

from typing import Any, Dict, Mapping

# All measurements that participate in the default ranking (10 values across 3 subgroups).
_ALL_MEASUREMENTS = [
    "EntityAlignmentMetric.recall",
    "TripleAlignmentMetric.recall",
    "EntityAlignmentMetric.precision",
    "TripleAlignmentMetric.precision",
    "DisjointDomainMetric.normalized_score",
    "DomainMetric.normalized_score",
    "RangeMetric.normalized_score",
    "DatatypeFormatMetric.normalized_score",
    "DatatypeMetric.normalized_score",
    "RelationDirectionMetric.normalized_score",
]

# Subgroup means, then equal-weight mean of the three subgroup scores.
DEFAULT_AGGREGATION_CONFIG: Dict[str, Any] = {
    "subgroups": {
        "coverage": {
            "measurements": [
                {"metric": "EntityAlignmentMetric", "measurement": "recall"},
                {"metric": "TripleAlignmentMetric", "measurement": "recall"},
            ],
            "aggregation": "mean",
        },
        "correctness": {
            "measurements": [
                "EntityAlignmentMetric.precision",
                "TripleAlignmentMetric.precision",
            ],
            "aggregation": "mean",
        },
        "consistency": {
            "measurements": [
                "DisjointDomainMetric.normalized_score",
                "DomainMetric.normalized_score",
                "RangeMetric.normalized_score",
                "DatatypeFormatMetric.normalized_score",
                "DatatypeMetric.normalized_score",
                "RelationDirectionMetric.normalized_score",
            ],
            "aggregation": "mean",
        },
    },
    "final": {
        "aggregation": "weighted_mean",
        "weights": {
            "coverage": 0.3333,
            "correctness": 0.3333,
            "consistency": 0.3333,
        },
    },
}

# CUSTOM AGGREGATION CONFIG
CUSTOM_AGGREGATION_CONFIG: Dict[str, Any] = {
    "subgroups": {
        "coverage_and_correctness": {
            "measurements": [
                "EntityAlignmentMetric.recall", "TripleAlignmentMetric.recall",
                "EntityAlignmentMetric.precision", "TripleAlignmentMetric.precision"
            ],
            "aggregation": "harmonic_mean",
        },
        "consistency_and_correctness": {
            "measurements": [
                "DisjointDomainMetric.normalized_score",
                "DomainMetric.normalized_score",
                "RangeMetric.normalized_score",
                "DatatypeFormatMetric.normalized_score",
                "DatatypeMetric.normalized_score",
                "RelationDirectionMetric.normalized_score",
            ],
            "aggregation": "harmonic_mean",
        },
    },
    "final": {
        "aggregation": "harmonic_mean",
    },
}


# Flat harmonic mean over all measurements (no subgroup intermediate scores).
FLAT_HMEAN_AGGREGATION_CONFIG: Dict[str, Any] = {
    "subgroups": {
        "all": {
            "measurements": list(_ALL_MEASUREMENTS),
            "aggregation": "harmonic_mean",
        },
    },
    "final": {
        "aggregation": "mean",
    },
}

AGGREGATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": DEFAULT_AGGREGATION_CONFIG,
    "flat_hmean": FLAT_HMEAN_AGGREGATION_CONFIG,
    "custom": CUSTOM_AGGREGATION_CONFIG,
}


def get_aggregation_config(name: str) -> Mapping[str, Any]:
    try:
        return AGGREGATION_CONFIGS[name]
    except KeyError as exc:
        known = ", ".join(sorted(AGGREGATION_CONFIGS))
        raise ValueError(f"Unknown rank aggregation {name!r}; choose one of: {known}") from exc
