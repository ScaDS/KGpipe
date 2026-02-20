"""
Visualization of parameter clustering results.

Produces static plots (PNG) summarising how extracted parameters group
across tools:
  - cluster size distribution
  - tool × cluster heatmap (cross-tool clusters)
  - 2-D embedding scatter (PCA, coloured by tool)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from ..clustering.models import ClusteringResult

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style="whitegrid", font_scale=0.9)


class ParameterVisualizer:
    """Generate static visualizations from a ``ClusteringResult``."""

    def __init__(self, result: ClusteringResult, output_dir: Path):
        self.result = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> list[Path]:
        """Run every visualization and return list of saved file paths."""
        paths: list[Path] = []
        if not self.result.clusters:
            logger.warning("No clusters to visualize")
            return paths

        paths.append(self.plot_cluster_sizes())
        paths.append(self.plot_tool_cluster_heatmap())
        paths.append(self.plot_embedding_scatter())
        logger.info(
            "Generated %d visualization(s) in %s", len(paths), self.output_dir
        )
        return paths

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_cluster_sizes(
        self, filename: str = "_viz_cluster_sizes.png"
    ) -> Path:
        """Horizontal bar chart of cluster sizes (top-30)."""
        clusters = sorted(self.result.clusters, key=lambda c: -c.size())[:30]
        labels = [
            f"[{c.cluster_id}] {c.label}" + (" ★" if c.is_cross_tool() else "")
            for c in clusters
        ]
        sizes = [c.size() for c in clusters]
        colors = [
            "#4c72b0" if c.is_cross_tool() else "#c0c0c0" for c in clusters
        ]

        fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)))
        ax.barh(range(len(labels)), sizes, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Number of parameters")
        ax.set_title(
            f"Cluster sizes (top {len(clusters)} of {self.result.n_clusters})"
        )
        # Legend for cross-tool marker
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor="#4c72b0", label="Cross-tool"),
                Patch(facecolor="#c0c0c0", label="Single tool"),
            ],
            loc="lower right",
        )
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved cluster size chart to %s", path)
        return path

    def plot_tool_cluster_heatmap(
        self, filename: str = "_viz_tool_heatmap.png"
    ) -> Path:
        """Heatmap of tools × clusters (cross-tool clusters only)."""
        import pandas as pd

        cross = self.result.cross_tool_clusters()
        if not cross:
            # Fall back to top-20 clusters if no cross-tool clusters
            cross = sorted(self.result.clusters, key=lambda c: -c.size())[:20]

        all_tools = sorted(
            {m.tool_name for c in cross for m in c.members}
        )
        cluster_labels = [f"[{c.cluster_id}] {c.label}" for c in cross]

        matrix = np.zeros((len(all_tools), len(cross)), dtype=int)
        for j, c in enumerate(cross):
            for m in c.members:
                i = all_tools.index(m.tool_name)
                matrix[i, j] += 1

        df = pd.DataFrame(matrix, index=all_tools, columns=cluster_labels)

        fig, ax = plt.subplots(
            figsize=(max(6, len(cross) * 0.6), max(3, len(all_tools) * 0.5))
        )
        sns.heatmap(
            df,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Parameters per tool × cluster (cross-tool clusters)")
        ax.set_ylabel("Tool")
        ax.set_xlabel("Cluster")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved tool×cluster heatmap to %s", path)
        return path

    def plot_embedding_scatter(
        self, filename: str = "_viz_embedding_scatter.png"
    ) -> Path:
        """2-D PCA scatter of parameter embeddings, coloured by tool."""
        from sklearn.decomposition import PCA

        # Collect all members across clusters
        all_members = [m for c in self.result.clusters for m in c.members]
        cluster_for_member = [
            c.cluster_id for c in self.result.clusters for m in c.members
        ]

        # Check if embeddings are present; if not, recompute them
        has_embeddings = any(m.embedding is not None for m in all_members)
        if not has_embeddings and all_members:
            logger.info("Embeddings not in clustering result — recomputing")
            from ..clustering.similarity import embed_parameters

            embed_parameters(all_members)

        # Collect embeddings and metadata
        embeddings = []
        tools = []
        names = []
        cluster_ids = []
        for cid, m in zip(cluster_for_member, all_members):
            if m.embedding is not None:
                embeddings.append(m.embedding)
                tools.append(m.tool_name)
                names.append(m.name)
                cluster_ids.append(cid)

        if len(embeddings) < 3:
            # Not enough points for meaningful 2-D projection
            logger.warning(
                "Too few embedded parameters (%d) for scatter plot",
                len(embeddings),
            )
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "Too few parameters for scatter plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            path = self.output_dir / filename
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path

        X = np.array(embeddings, dtype=np.float32)
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        unique_tools = sorted(set(tools))
        palette = sns.color_palette("husl", len(unique_tools))
        tool_to_color = dict(zip(unique_tools, palette))

        fig, ax = plt.subplots(figsize=(10, 7))
        for tool in unique_tools:
            mask = [t == tool for t in tools]
            pts = X_2d[mask]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                label=tool,
                color=tool_to_color[tool],
                alpha=0.65,
                s=30,
                edgecolors="white",
                linewidth=0.3,
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title(
            f"Parameter embeddings — {self.result.n_parameters} params, "
            f"{self.result.n_clusters} clusters"
        )
        ax.legend(title="Tool", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved embedding scatter to %s", path)
        return path

    # ------------------------------------------------------------------
    # Alternate constructor from JSON file
    # ------------------------------------------------------------------

    @classmethod
    def from_clusters_json(
        cls, json_path: Path, output_dir: Optional[Path] = None
    ) -> "ParameterVisualizer":
        """
        Create a visualizer from a ``_clusters.json`` file.

        If *output_dir* is not given, plots are saved next to the JSON file.
        """
        import json

        with open(json_path) as f:
            data = json.load(f)

        result = ClusteringResult.model_validate(data)
        return cls(result, output_dir or json_path.parent)
