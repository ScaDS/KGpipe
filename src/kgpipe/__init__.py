"""
KGbench - Knowledge Graph Benchmarking Framework

A framework for generating and executing KG pipelines with evaluation capabilities.
"""

__version__ = "0.6.0"
__author__ = "KGflex Team"

# Import main components for easy access
from .common.models import (
    Data, DataFormat, KgTask, KgTaskReport, DataSet, KG, 
    Stage, Metric, EvaluationReport, KgPipe
)

# from .common.decorators import flextask, MetaKg

__all__ = [
    # Models
    "Data", "DataFormat", "KgTask", "KgTaskReport", "DataSet", "KG",
    "Stage", "Metric", "EvaluationReport", "KgPipe",
    # Decorators
    # "flextask", "MetaKg",
    # Version
    "__version__",
] 