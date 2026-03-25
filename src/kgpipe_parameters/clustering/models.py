"""
Data models for parameter clustering results.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
import numpy as np


class ParameterVector(BaseModel):
    """A parameter together with its embedding and origin metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Normalized parameter name")
    tool_name: str = Field(..., description="Tool this parameter belongs to")
    native_keys: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    type_hint: Optional[str] = None
    default_value: Optional[Any] = None
    required: bool = False
    source_label: str = Field(
        "", description="Human-readable source (e.g. 'cli', 'readme:README.md')"
    )
    # Embedding stored as plain list for JSON serialisation; converted to
    # numpy array for computation.
    embedding: Optional[List[float]] = Field(
        None, description="Sentence-transformer embedding"
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def text_for_embedding(self) -> str:
        """Build the text representation used for embedding computation."""
        parts = [self.name.replace("_", " ")]
        if self.description:
            parts.append(self.description)
        if self.native_keys:
            parts.append(" ".join(self.native_keys))
        if self.type_hint:
            parts.append(f"type: {self.type_hint}")
        return " | ".join(parts)


class ParameterCluster(BaseModel):
    """A cluster of similar parameters found across one or more tools."""

    cluster_id: int = Field(..., description="Numeric cluster identifier")
    label: str = Field(
        "", description="Human-readable label (e.g. most common parameter name)"
    )
    members: List[ParameterVector] = Field(default_factory=list)
    tools: List[str] = Field(
        default_factory=list,
        description="Distinct tool names represented in this cluster",
    )
    centroid: Optional[List[float]] = Field(
        None, description="Mean embedding of the cluster"
    )

    def size(self) -> int:
        return len(self.members)

    def is_cross_tool(self) -> bool:
        """Return True if parameters from more than one tool are in this cluster."""
        return len(self.tools) > 1


class ClusteringResult(BaseModel):
    """Container for an entire clustering run."""

    n_parameters: int = Field(0, description="Total parameters fed to clustering")
    n_clusters: int = Field(0, description="Number of clusters produced")
    distance_threshold: float = Field(
        0.0, description="Distance threshold used for clustering"
    )
    model_name: str = Field("", description="Sentence-transformer model used")
    clusters: List[ParameterCluster] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def cross_tool_clusters(self) -> List[ParameterCluster]:
        """Return only clusters that span more than one tool."""
        return [c for c in self.clusters if c.is_cross_tool()]

    def to_table_rows(self) -> List[Dict[str, Any]]:
        """
        Flatten clusters into a list of rows suitable for a pandas DataFrame
        or CSV export.
        """
        rows: List[Dict[str, Any]] = []
        for cluster in self.clusters:
            for member in cluster.members:
                rows.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "cluster_label": cluster.label,
                        "cluster_size": cluster.size(),
                        "cross_tool": cluster.is_cross_tool(),
                        "tool": member.tool_name,
                        "parameter": member.name,
                        "native_keys": ", ".join(member.native_keys),
                        "description": member.description or "",
                        "type_hint": member.type_hint or "",
                        "default_value": member.default_value,
                        "required": member.required,
                        "source": member.source_label,
                    }
                )
        return rows

