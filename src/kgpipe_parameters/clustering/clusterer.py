"""
Main clustering logic.

Loads extracted parameters from experiment JSON output, embeds them with
sentence-transformers, and applies agglomerative clustering to surface
groups of similar configuration knobs across tools.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .models import ParameterVector, ParameterCluster, ClusteringResult
from .similarity import DEFAULT_MODEL_NAME, embed_parameters

logger = logging.getLogger(__name__)


class ParameterClusterer:
    """
    Cluster extracted parameters by semantic similarity.

    Typical usage::

        clusterer = ParameterClusterer()
        result = clusterer.cluster_from_output_dir(Path("output/"))
        for c in result.cross_tool_clusters():
            print(c.label, c.tools, c.size())
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        distance_threshold: float = 0.55,
        min_cluster_size: int = 1,
    ):
        """
        Parameters
        ----------
        model_name : str
            Sentence-transformer model to use for embeddings.
        distance_threshold : float
            Maximum cosine *distance* (1 − similarity) at which two
            parameters are still merged into the same cluster.
            Lower → tighter clusters.  ``0.55`` is a good starting
            point for short technical phrases.
        min_cluster_size : int
            Drop clusters smaller than this after clustering.
        """
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_parameters_from_json(path: Path) -> List[ParameterVector]:
        """
        Load parameters from one tool's JSON output file.

        Expected format: the JSON written by
        ``ToolExtractionResult.to_dict()`` — a dict with a
        ``"parameters"`` list and a ``"tool_name"`` string.
        """
        with open(path) as f:
            data = json.load(f)

        tool_name = data.get("tool_name", path.stem)
        vectors: List[ParameterVector] = []

        for p in data.get("parameters", []):
            pv = ParameterVector(
                name=p.get("name", ""),
                tool_name=tool_name,
                native_keys=p.get("native_keys", []),
                description=p.get("description"),
                type_hint=p.get("type_hint"),
                default_value=p.get("default_value"),
                required=p.get("required", False),
                source_label=p.get("_source", ""),
            )
            vectors.append(pv)

        return vectors

    def load_from_output_dir(self, output_dir: Path) -> List[ParameterVector]:
        """
        Load parameters from *all* tool JSON files in *output_dir*.

        Skips files whose name starts with ``_`` (e.g. ``_summary.json``).
        """
        all_params: List[ParameterVector] = []
        for json_file in sorted(output_dir.glob("*.json")):
            if json_file.name.startswith("_"):
                continue
            try:
                params = self.load_parameters_from_json(json_file)
                logger.info(
                    "Loaded %d parameters from %s", len(params), json_file.name
                )
                all_params.extend(params)
            except Exception as e:
                logger.warning("Failed to load %s: %s", json_file, e)

        logger.info("Total parameters loaded: %d", len(all_params))
        return all_params

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster(self, parameters: List[ParameterVector]) -> ClusteringResult:
        """
        Embed and cluster a list of parameters.

        Returns a ``ClusteringResult`` with numbered clusters.
        """
        if not parameters:
            return ClusteringResult(
                model_name=self.model_name,
                distance_threshold=self.distance_threshold,
            )

        # 1. Compute embeddings
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        embeddings = embed_parameters(
            parameters, model_name=self.model_name, model=self._model
        )

        # 2. Agglomerative clustering with cosine distance
        n = len(parameters)

        if n == 1:
            # AgglomerativeClustering requires ≥ 2 samples; short-circuit.
            labels = np.array([0])
        else:
            from sklearn.cluster import AgglomerativeClustering

            sim_matrix = embeddings @ embeddings.T
            np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)
            dist_matrix = 1.0 - sim_matrix

            clustering_model = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=self.distance_threshold,
            )
            labels = clustering_model.fit_predict(dist_matrix)

        # 3. Build ParameterCluster objects
        cluster_map: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            cluster_map.setdefault(int(label), []).append(idx)

        clusters: List[ParameterCluster] = []
        for cid, member_indices in sorted(cluster_map.items()):
            members = [parameters[i] for i in member_indices]
            if len(members) < self.min_cluster_size:
                continue

            tools = sorted(set(m.tool_name for m in members))
            centroid = embeddings[member_indices].mean(axis=0)

            # Label = most common parameter name in the cluster
            name_counts = Counter(m.name for m in members)
            label_name = name_counts.most_common(1)[0][0]

            clusters.append(
                ParameterCluster(
                    cluster_id=cid,
                    label=label_name,
                    members=members,
                    tools=tools,
                    centroid=centroid.tolist(),
                )
            )

        # Sort: cross-tool first, then by size descending
        clusters.sort(key=lambda c: (-int(c.is_cross_tool()), -c.size()))

        return ClusteringResult(
            n_parameters=len(parameters),
            n_clusters=len(clusters),
            distance_threshold=self.distance_threshold,
            model_name=self.model_name,
            clusters=clusters,
        )

    def cluster_from_output_dir(self, output_dir: Path) -> ClusteringResult:
        """Convenience: load + cluster in one call."""
        params = self.load_from_output_dir(output_dir)
        return self.cluster(params)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_result(result: ClusteringResult, path: Path) -> None:
        """Save clustering result as JSON."""
        # Strip large embedding lists to keep the file readable
        data = result.model_dump()
        for cluster in data.get("clusters", []):
            cluster.pop("centroid", None)
            for member in cluster.get("members", []):
                member.pop("embedding", None)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved clustering result to %s", path)

    @staticmethod
    def save_table(result: ClusteringResult, path: Path) -> None:
        """Save a flat CSV parameter table from clustering results."""
        import csv

        rows = result.to_table_rows()
        if not rows:
            logger.warning("No rows to write to table")
            return

        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved parameter table (%d rows) to %s", len(rows), path)

