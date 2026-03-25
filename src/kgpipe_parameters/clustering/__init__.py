"""
Parameter clustering module.

Groups similar parameters across tools using sentence-transformer embeddings
and agglomerative clustering so that common configuration knobs are surfaced.
"""

from .models import ParameterVector, ParameterCluster, ClusteringResult
from .similarity import embed_parameters, cosine_similarity_matrix
from .clusterer import ParameterClusterer

__all__ = [
    "ParameterVector",
    "ParameterCluster",
    "ClusteringResult",
    "embed_parameters",
    "cosine_similarity_matrix",
    "ParameterClusterer",
]

