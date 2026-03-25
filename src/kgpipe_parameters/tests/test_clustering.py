"""
Tests for the parameter clustering module.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List

from kgpipe_parameters.clustering.models import (
    ParameterVector,
    ParameterCluster,
    ClusteringResult,
)
from kgpipe_parameters.clustering.similarity import (
    embed_parameters,
    cosine_similarity_matrix,
)
from kgpipe_parameters.clustering.clusterer import ParameterClusterer


# ============================================================================
# Fixtures
# ============================================================================


def _make_param(
    name: str,
    tool: str,
    description: str = "",
    native_keys: List[str] | None = None,
    type_hint: str | None = None,
    default_value=None,
) -> ParameterVector:
    return ParameterVector(
        name=name,
        tool_name=tool,
        native_keys=native_keys or [],
        description=description,
        type_hint=type_hint,
        default_value=default_value,
        source_label=f"{tool}/source",
    )


@pytest.fixture
def sample_parameters() -> List[ParameterVector]:
    """A small set of parameters from two fictitious tools."""
    return [
        # Tool A
        _make_param("threshold", "tool_a", "Matching threshold value", ["--threshold", "-t"], "float", 0.5),
        _make_param("max_iterations", "tool_a", "Maximum number of iterations", ["--max-iter"], "int", 100),
        _make_param("output_dir", "tool_a", "Output directory path", ["--output", "-o"], "str"),
        _make_param("batch_size", "tool_a", "Number of items per batch", ["--batch-size"], "int", 32),
        # Tool B
        _make_param("similarity_threshold", "tool_b", "Threshold for similarity matching", ["--sim-threshold"], "float", 0.7),
        _make_param("iterations", "tool_b", "Number of iterations to run", ["--iterations", "-n"], "int", 50),
        _make_param("output_path", "tool_b", "Path for output files", ["--output-path"], "str"),
        _make_param("learning_rate", "tool_b", "Learning rate for optimizer", ["--lr"], "float", 0.001),
    ]


@pytest.fixture
def mock_sentence_model():
    """A mock SentenceTransformer that returns deterministic embeddings."""
    model = MagicMock()
    # Return embeddings designed so that similar parameters are closer.
    # Each "encode" call gets a list of texts; we return a (N, 8) array
    # seeded from the text hash so it is deterministic.

    def _encode(texts, batch_size=64, show_progress_bar=False):
        rng = np.random.RandomState(42)
        # Use a small embedding dim for testing speed
        embs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 2**31
            r = np.random.RandomState(seed)
            embs.append(r.randn(8).astype(np.float32))
        return np.array(embs)

    model.encode = _encode
    return model


# ============================================================================
# ParameterVector tests
# ============================================================================


class TestParameterVector:
    def test_text_for_embedding_basic(self):
        pv = _make_param("threshold", "t", "matching threshold", ["--threshold"])
        text = pv.text_for_embedding()
        assert "threshold" in text
        assert "matching threshold" in text
        assert "--threshold" in text

    def test_text_for_embedding_minimal(self):
        pv = _make_param("x", "t")
        text = pv.text_for_embedding()
        assert "x" in text


# ============================================================================
# ParameterCluster tests
# ============================================================================


class TestParameterCluster:
    def test_size(self):
        members = [_make_param("a", "t1"), _make_param("b", "t2")]
        cluster = ParameterCluster(cluster_id=0, label="a", members=members, tools=["t1", "t2"])
        assert cluster.size() == 2

    def test_is_cross_tool(self):
        c1 = ParameterCluster(cluster_id=0, label="x", members=[], tools=["t1", "t2"])
        assert c1.is_cross_tool()

        c2 = ParameterCluster(cluster_id=1, label="x", members=[], tools=["t1"])
        assert not c2.is_cross_tool()


# ============================================================================
# ClusteringResult tests
# ============================================================================


class TestClusteringResult:
    def test_cross_tool_clusters(self):
        c1 = ParameterCluster(cluster_id=0, label="x", members=[], tools=["t1", "t2"])
        c2 = ParameterCluster(cluster_id=1, label="y", members=[], tools=["t1"])
        result = ClusteringResult(clusters=[c1, c2], n_clusters=2)
        assert len(result.cross_tool_clusters()) == 1

    def test_to_table_rows(self):
        members = [_make_param("threshold", "t1"), _make_param("threshold", "t2")]
        cluster = ParameterCluster(cluster_id=0, label="threshold", members=members, tools=["t1", "t2"])
        result = ClusteringResult(clusters=[cluster], n_clusters=1, n_parameters=2)
        rows = result.to_table_rows()
        assert len(rows) == 2
        assert rows[0]["cluster_label"] == "threshold"
        assert rows[0]["tool"] == "t1"
        assert rows[1]["tool"] == "t2"

    def test_to_table_rows_empty(self):
        result = ClusteringResult()
        assert result.to_table_rows() == []


# ============================================================================
# Similarity tests
# ============================================================================


class TestSimilarity:
    def test_embed_parameters(self, sample_parameters, mock_sentence_model):
        embeddings = embed_parameters(
            sample_parameters, model=mock_sentence_model
        )
        assert embeddings.shape[0] == len(sample_parameters)
        assert embeddings.shape[1] > 0
        # All embeddings should be stored back
        for pv in sample_parameters:
            assert pv.embedding is not None
            assert len(pv.embedding) == embeddings.shape[1]

    def test_embed_parameters_empty(self, mock_sentence_model):
        embeddings = embed_parameters([], model=mock_sentence_model)
        assert embeddings.shape == (0, 0)

    def test_cosine_similarity_matrix_identity(self):
        embs = np.eye(3, dtype=np.float32)
        sim = cosine_similarity_matrix(embs)
        np.testing.assert_allclose(sim, np.eye(3), atol=1e-5)

    def test_cosine_similarity_matrix_same_vector(self):
        embs = np.ones((4, 5), dtype=np.float32)
        sim = cosine_similarity_matrix(embs)
        np.testing.assert_allclose(sim, np.ones((4, 4)), atol=1e-5)

    def test_cosine_similarity_matrix_empty(self):
        embs = np.empty((0, 0))
        sim = cosine_similarity_matrix(embs)
        assert sim.shape == (0, 0)


# ============================================================================
# ParameterClusterer tests
# ============================================================================


class TestParameterClusterer:
    def test_cluster_basic(self, sample_parameters, mock_sentence_model):
        """Clustering should produce at least one cluster."""
        clusterer = ParameterClusterer(distance_threshold=0.8)
        clusterer._model = mock_sentence_model

        result = clusterer.cluster(sample_parameters)
        assert result.n_parameters == len(sample_parameters)
        assert result.n_clusters > 0
        # All parameters should be assigned to some cluster
        total_members = sum(c.size() for c in result.clusters)
        assert total_members == len(sample_parameters)

    def test_cluster_empty(self):
        clusterer = ParameterClusterer()
        result = clusterer.cluster([])
        assert result.n_parameters == 0
        assert result.n_clusters == 0

    def test_cluster_single_param(self, mock_sentence_model):
        clusterer = ParameterClusterer()
        clusterer._model = mock_sentence_model
        params = [_make_param("threshold", "tool_a", "test")]
        result = clusterer.cluster(params)
        assert result.n_parameters == 1
        assert result.n_clusters == 1

    def test_load_parameters_from_json(self, tmp_path):
        """Test loading parameters from a tool JSON output file."""
        data = {
            "tool_name": "test_tool",
            "parameters": [
                {
                    "name": "threshold",
                    "native_keys": ["--threshold"],
                    "description": "test",
                    "type_hint": "float",
                    "default_value": 0.5,
                    "required": False,
                    "_source": "cli",
                },
                {
                    "name": "output",
                    "native_keys": ["--output"],
                    "description": "output path",
                    "_source": "cli",
                },
            ],
        }
        json_file = tmp_path / "test_tool.json"
        json_file.write_text(json.dumps(data))

        params = ParameterClusterer.load_parameters_from_json(json_file)
        assert len(params) == 2
        assert params[0].name == "threshold"
        assert params[0].tool_name == "test_tool"
        assert params[1].name == "output"

    def test_load_from_output_dir(self, tmp_path):
        """Test loading from a directory with multiple tool files."""
        for tool_name in ["tool_a", "tool_b"]:
            data = {
                "tool_name": tool_name,
                "parameters": [
                    {"name": "param1", "native_keys": [], "_source": "cli"},
                ],
            }
            (tmp_path / f"{tool_name}.json").write_text(json.dumps(data))
        # Summary file should be skipped
        (tmp_path / "_summary.json").write_text("{}")

        clusterer = ParameterClusterer()
        params = clusterer.load_from_output_dir(tmp_path)
        assert len(params) == 2
        tool_names = {p.tool_name for p in params}
        assert tool_names == {"tool_a", "tool_b"}

    def test_save_result(self, tmp_path):
        members = [_make_param("threshold", "t1")]
        cluster = ParameterCluster(cluster_id=0, label="threshold", members=members, tools=["t1"])
        result = ClusteringResult(clusters=[cluster], n_clusters=1, n_parameters=1)

        out_path = tmp_path / "clusters.json"
        ParameterClusterer.save_result(result, out_path)
        assert out_path.exists()

        saved = json.loads(out_path.read_text())
        assert saved["n_clusters"] == 1
        assert len(saved["clusters"]) == 1

    def test_save_table(self, tmp_path):
        members = [
            _make_param("threshold", "t1", description="test"),
            _make_param("threshold", "t2", description="test"),
        ]
        cluster = ParameterCluster(cluster_id=0, label="threshold", members=members, tools=["t1", "t2"])
        result = ClusteringResult(clusters=[cluster], n_clusters=1, n_parameters=2)

        csv_path = tmp_path / "table.csv"
        ParameterClusterer.save_table(result, csv_path)
        assert csv_path.exists()

        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["parameter"] == "threshold"

    def test_cluster_from_output_dir(self, tmp_path, mock_sentence_model):
        """Integration test: load → cluster from an output directory."""
        for tool_name, params in [
            ("tool_a", [
                {"name": "threshold", "native_keys": ["--threshold"], "description": "match threshold", "_source": "cli"},
                {"name": "output", "native_keys": ["--output"], "description": "output path", "_source": "cli"},
            ]),
            ("tool_b", [
                {"name": "similarity_threshold", "native_keys": ["--sim-threshold"], "description": "threshold for similarity", "_source": "cli"},
                {"name": "output_dir", "native_keys": ["--output-dir"], "description": "directory for output", "_source": "cli"},
            ]),
        ]:
            data = {"tool_name": tool_name, "parameters": params}
            (tmp_path / f"{tool_name}.json").write_text(json.dumps(data))

        clusterer = ParameterClusterer(distance_threshold=0.8)
        clusterer._model = mock_sentence_model
        result = clusterer.cluster_from_output_dir(tmp_path)

        assert result.n_parameters == 4
        assert result.n_clusters > 0

