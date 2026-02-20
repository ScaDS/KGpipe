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
    ReadmeDocExtractor,
    LLMReadmeDocExtractor,
)

from .clustering import (
    ParameterClusterer,
    ParameterVector,
    ParameterCluster,
    ClusteringResult,
)

from .visualization import ParameterVisualizer

__all__ = [
    # Extraction
    "ParameterMiner",
    "RawParameter",
    "ExtractionResult",
    "SourceType",
    "ExtractionMethod",
    "ReadmeDocExtractor",
    "LLMReadmeDocExtractor",
    # Clustering
    "ParameterClusterer",
    "ParameterVector",
    "ParameterCluster",
    "ClusteringResult",
    # Visualization
    "ParameterVisualizer",
]

