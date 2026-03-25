"""
Tests for the parameter visualization module.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import List

from kgpipe_parameters.clustering.models import (
    ParameterVector,
    ParameterCluster,
    ClusteringResult,
)
from kgpipe_parameters.visualization import ParameterVisualizer


# ============================================================================
# Helpers
# ============================================================================


def _make_param(
    name: str,
    tool: str,
    description: str = "",
    embedding: List[float] | None = None,
) -> ParameterVector:
    return ParameterVector(
        name=name,
        tool_name=tool,
        description=description,
        native_keys=[f"--{name}"],
        source_label=f"{tool}/source",
        embedding=embedding,
    )


def _random_embedding(
    dim: int = 16, rng: np.random.Generator | None = None
) -> List[float]:
    rng = rng or np.random.default_rng(42)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def sample_clustering_result() -> ClusteringResult:
    """A small synthetic clustering result for visualization tests."""
    rng = np.random.default_rng(0)

    # Cluster 0: cross-tool (threshold, 3 params, 2 tools)
    c0_members = [
        _make_param(
            "threshold", "tool_a", "Matching threshold", _random_embedding(rng=rng)
        ),
        _make_param(
            "threshold", "tool_b", "Score threshold", _random_embedding(rng=rng)
        ),
        _make_param(
            "similarity_threshold",
            "tool_a",
            "Similarity cutoff",
            _random_embedding(rng=rng),
        ),
    ]
    c0 = ParameterCluster(
        cluster_id=0,
        label="threshold",
        members=c0_members,
        tools=["tool_a", "tool_b"],
    )

    # Cluster 1: cross-tool (output, 2 params, 2 tools)
    c1_members = [
        _make_param(
            "output_dir", "tool_a", "Output directory", _random_embedding(rng=rng)
        ),
        _make_param(
            "output_path", "tool_b", "Output file path", _random_embedding(rng=rng)
        ),
    ]
    c1 = ParameterCluster(
        cluster_id=1,
        label="output_dir",
        members=c1_members,
        tools=["tool_a", "tool_b"],
    )

    # Cluster 2: single-tool (verbose, 2 params)
    c2_members = [
        _make_param(
            "verbose", "tool_a", "Verbosity level", _random_embedding(rng=rng)
        ),
        _make_param("debug", "tool_a", "Debug mode", _random_embedding(rng=rng)),
    ]
    c2 = ParameterCluster(
        cluster_id=2,
        label="verbose",
        members=c2_members,
        tools=["tool_a"],
    )

    return ClusteringResult(
        n_parameters=7,
        n_clusters=3,
        distance_threshold=0.55,
        model_name="test-model",
        clusters=[c0, c1, c2],
    )


# ============================================================================
# Tests
# ============================================================================


class TestParameterVisualizer:
    """Tests for ParameterVisualizer."""

    def test_generate_all_creates_files(self, sample_clustering_result, tmp_path):
        viz = ParameterVisualizer(sample_clustering_result, tmp_path)
        paths = viz.generate_all()
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_plot_cluster_sizes(self, sample_clustering_result, tmp_path):
        viz = ParameterVisualizer(sample_clustering_result, tmp_path)
        path = viz.plot_cluster_sizes()
        assert path.exists()
        assert path.name == "_viz_cluster_sizes.png"

    def test_plot_tool_cluster_heatmap(self, sample_clustering_result, tmp_path):
        viz = ParameterVisualizer(sample_clustering_result, tmp_path)
        path = viz.plot_tool_cluster_heatmap()
        assert path.exists()
        assert path.name == "_viz_tool_heatmap.png"

    def test_plot_embedding_scatter(self, sample_clustering_result, tmp_path):
        viz = ParameterVisualizer(sample_clustering_result, tmp_path)
        path = viz.plot_embedding_scatter()
        assert path.exists()
        assert path.name == "_viz_embedding_scatter.png"

    def test_empty_result_returns_empty(self, tmp_path):
        empty = ClusteringResult()
        viz = ParameterVisualizer(empty, tmp_path)
        paths = viz.generate_all()
        assert paths == []

    def test_from_clusters_json(self, sample_clustering_result, tmp_path):
        # Write a JSON file
        json_path = tmp_path / "_clusters.json"
        data = sample_clustering_result.model_dump()
        # Strip centroids/embeddings like the real save does
        for c in data.get("clusters", []):
            c.pop("centroid", None)
        with open(json_path, "w") as f:
            json.dump(data, f, default=str)

        viz = ParameterVisualizer.from_clusters_json(json_path)
        assert viz.result.n_clusters == 3

    def test_scatter_too_few_points(self, tmp_path):
        """Scatter plot gracefully handles < 3 embedded parameters."""
        m = _make_param("x", "t", embedding=_random_embedding())
        c = ParameterCluster(cluster_id=0, label="x", members=[m], tools=["t"])
        result = ClusteringResult(n_parameters=1, n_clusters=1, clusters=[c])
        viz = ParameterVisualizer(result, tmp_path)
        path = viz.plot_embedding_scatter()
        assert path.exists()

