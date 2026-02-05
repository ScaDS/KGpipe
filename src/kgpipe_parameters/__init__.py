"""
KGpipe Parameters subpackage for analyzing and optimizing parameters for data integration tasks.

This package provides functionality to:
1. Extract/Find configuration Parameters for a Task T and its implementations I
2. Match and cluster configuration parameters
3. Find best configuration parameters
"""

from .extraction import (
    ParameterMiner,
    RawParameter,
    ExtractionResult,
    SourceType,
    ExtractionMethod,
)

__all__ = [
    "ParameterMiner",
    "RawParameter",
    "ExtractionResult",
    "SourceType",
    "ExtractionMethod",
]

