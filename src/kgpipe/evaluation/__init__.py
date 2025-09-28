"""
KGbench Evaluation Module

This module provides comprehensive evaluation capabilities for knowledge graphs
across three main aspects: statistical metrics, semantic evaluation, and reference-based evaluation.
"""

from .evaluator import Evaluator, EvaluationConfig
from .base import EvaluationAspect
from .metrics import Metric, MetricResult
from .reports import EvaluationReport

__all__ = [
    "Evaluator",
    "EvaluationConfig", 
    "EvaluationAspect",
    "Metric",
    "MetricResult",
    "EvaluationReport"
]
