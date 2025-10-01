"""
Metrics Module

Base classes and types for evaluation metrics.
"""

from ..base import Metric, MetricResult



"""
Metrics Registration Module

This module registers all evaluation metrics for discovery by the CLI.
"""

from kgpipe.evaluation.aspects.statistical import (
    EntityCountMetric, RelationCountMetric, TripleCountMetric, ClassCountMetric
)
from kgpipe.evaluation.aspects.semantic import (
    ReasoningMetric, SchemaConsistencyMetric, NamespaceUsageMetric
)
from kgpipe.evaluation.aspects.reference import (
    ER_EntityMatchMetric, ER_RelationMatchMetric
)


__all__ = [
    "Metric", 
    "MetricResult",
    "EntityCountMetric", 
    "RelationCountMetric", 
    "TripleCountMetric", 
    "ClassCountMetric",
    "ReasoningMetric", 
    "SchemaConsistencyMetric", 
    "NamespaceUsageMetric",
    "ER_EntityMatchMetric", 
    "ER_RelationMatchMetric"
]