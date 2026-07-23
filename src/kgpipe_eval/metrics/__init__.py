from .statistics import CountMetric
from .triple_alignment import TripleAlignmentMetric
from .entity_alignment import EntityAlignmentMetric
from .duplicates import DuplicateMetric
from .consistency_violations import (
    DisjointDomainMetric,
    DomainMetric,
    RangeMetric,
    RelationDirectionMetric,
    DatatypeMetric,
    DatatypeFormatMetric,
)
from kgpipe_eval.api import Metric

# Catalog of kgpipe_eval metrics used by Registry discovery.
METRIC_CLASSES: tuple[type[Metric], ...] = (
    CountMetric,
    DuplicateMetric,
    EntityAlignmentMetric,
    TripleAlignmentMetric,
    DisjointDomainMetric,
    DomainMetric,
    RangeMetric,
    RelationDirectionMetric,
    DatatypeMetric,
    DatatypeFormatMetric,
)


def register_metrics() -> None:
    """Register all kgpipe_eval metrics into the global Registry."""
    from kgpipe.common.registry import Registry

    for metric_cls in METRIC_CLASSES:
        Registry.add_metric(metric_cls)


__all__ = [
    "CountMetric",
    "TripleAlignmentMetric",
    "EntityAlignmentMetric",
    "DuplicateMetric",
    "DisjointDomainMetric",
    "DomainMetric",
    "RangeMetric",
    "RelationDirectionMetric",
    "DatatypeMetric",
    "DatatypeFormatMetric",
    "METRIC_CLASSES",
    "register_metrics",
]
