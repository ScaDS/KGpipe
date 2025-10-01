"""
Provenance Tracking for KGbench

This module provides classes for tracking and managing provenance information
as a meta knowledge graph, including execution tracking, data lineage,
and registry export capabilities.
"""

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from pathlib import Path
import json
import hashlib

from .ontology import KGBENCH_NAMESPACE, PROV_NAMESPACE, RDFS_NAMESPACE
from .vocabularies import (
    get_execution_status_iri,
    get_data_format_iri,
    get_task_category_iri,
    get_metric_type_iri
)


@dataclass
class ProvenanceNode:
    """A node in the provenance graph."""
    iri: str
    node_type: str  # Task, Pipeline, Execution, Data, Metric, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_property(self, key: str, value: Any) -> None:
        """Update a property and set updated_at timestamp."""
        self.properties[key] = value
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "iri": self.iri,
            "type": self.node_type,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class ProvenanceEdge:
    """An edge in the provenance graph."""
    source_iri: str
    target_iri: str
    edge_type: str  # wasGeneratedBy, used, wasDerivedFrom, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "source": self.source_iri,
            "target": self.target_iri,
            "type": self.edge_type,
            "properties": self.properties,
            "created_at": self.created_at.isoformat()
        }


class ProvenanceGraph:
    """A graph representing provenance information."""
    
    def __init__(self, name: str = "kgpipe_provenance"):
        self.name = name
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[ProvenanceEdge] = []
        self.created_at = datetime.now()
    
    def add_node(self, node: ProvenanceNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.iri] = node
    
    def get_node(self, iri: str) -> Optional[ProvenanceNode]:
        """Get a node by IRI."""
        return self.nodes.get(iri)
    
    def add_edge(self, edge: ProvenanceEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
    
    def get_edges(self, source_iri: Optional[str] = None, 
                  target_iri: Optional[str] = None,
                  edge_type: Optional[str] = None) -> List[ProvenanceEdge]:
        """Get edges with optional filtering."""
        filtered_edges = []
        for edge in self.edges:
            if source_iri and edge.source_iri != source_iri:
                continue
            if target_iri and edge.target_iri != target_iri:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            filtered_edges.append(edge)
        return filtered_edges
    
    def get_neighbors(self, node_iri: str, direction: str = "both") -> List[str]:
        """Get neighbor IRIs for a node."""
        neighbors = set()
        for edge in self.edges:
            if direction in ["both", "out"] and edge.source_iri == node_iri:
                neighbors.add(edge.target_iri)
            if direction in ["both", "in"] and edge.target_iri == node_iri:
                neighbors.add(edge.source_iri)
        return list(neighbors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "nodes": {iri: node.to_dict() for iri, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges]
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the graph to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ProvenanceGraph':
        """Load a graph from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = cls(data["name"])
        graph.created_at = datetime.fromisoformat(data["created_at"])
        
        # Load nodes
        for iri, node_data in data["nodes"].items():
            node = ProvenanceNode(
                iri=node_data["iri"],
                node_type=node_data["type"],
                properties=node_data["properties"],
                created_at=datetime.fromisoformat(node_data["created_at"]),
                updated_at=datetime.fromisoformat(node_data["updated_at"])
            )
            graph.add_node(node)
        
        # Load edges
        for edge_data in data["edges"]:
            edge = ProvenanceEdge(
                source_iri=edge_data["source"],
                target_iri=edge_data["target"],
                edge_type=edge_data["type"],
                properties=edge_data["properties"],
                created_at=datetime.fromisoformat(edge_data["created_at"])
            )
            graph.add_edge(edge)
        
        return graph


class ExecutionTracker:
    """Tracks execution provenance for tasks and pipelines."""
    
    def __init__(self, graph: ProvenanceGraph):
        self.graph = graph
        self.current_execution_id: Optional[str] = None
    
    def start_execution(self, execution_id: str, pipeline_iri: str, 
                       configuration: Dict[str, Any]) -> None:
        """Start tracking an execution."""
        self.current_execution_id = execution_id
        
        # Create execution node
        execution_node = ProvenanceNode(
            iri=execution_id,
            node_type=f"{KGBENCH_NAMESPACE}Execution",
            properties={
                f"{PROV_NAMESPACE}startedAtTime": datetime.now().isoformat(),
                f"{KGBENCH_NAMESPACE}hasStatus": get_execution_status_iri("RUNNING"),
                f"{KGBENCH_NAMESPACE}hasConfiguration": json.dumps(configuration)
            }
        )
        self.graph.add_node(execution_node)
        
        # Link to pipeline
        self.graph.add_edge(ProvenanceEdge(
            source_iri=execution_id,
            target_iri=pipeline_iri,
            edge_type=f"{PROV_NAMESPACE}wasInformedBy"
        ))
    
    def add_task_execution(self, task_id: str, task_iri: str, 
                          inputs: List[str], outputs: List[str],
                          metrics: Dict[str, Any]) -> None:
        """Add a task execution to the current execution."""
        if not self.current_execution_id:
            raise ValueError("No current execution active")
        
        # Create task execution node
        task_exec_node = ProvenanceNode(
            iri=task_id,
            node_type=f"{KGBENCH_NAMESPACE}TaskExecution",
            properties={
                f"{PROV_NAMESPACE}startedAtTime": datetime.now().isoformat(),
                f"{KGBENCH_NAMESPACE}hasMetrics": json.dumps(metrics)
            }
        )
        self.graph.add_node(task_exec_node)
        
        # Link to task definition
        self.graph.add_edge(ProvenanceEdge(
            source_iri=task_id,
            target_iri=task_iri,
            edge_type=f"{PROV_NAMESPACE}wasInformedBy"
        ))
        
        # Link to parent execution
        self.graph.add_edge(ProvenanceEdge(
            source_iri=self.current_execution_id,
            target_iri=task_id,
            edge_type=f"{PROV_NAMESPACE}wasAssociatedWith"
        ))
        
        # Link inputs
        for input_iri in inputs:
            self.graph.add_edge(ProvenanceEdge(
                source_iri=task_id,
                target_iri=input_iri,
                edge_type=f"{PROV_NAMESPACE}used"
            ))
        
        # Link outputs
        for output_iri in outputs:
            self.graph.add_edge(ProvenanceEdge(
                source_iri=output_iri,
                target_iri=task_id,
                edge_type=f"{PROV_NAMESPACE}wasGeneratedBy"
            ))
    
    def end_execution(self, status: str = "COMPLETED", 
                     exit_code: int = 0, logs: str = "") -> None:
        """End the current execution."""
        if not self.current_execution_id:
            raise ValueError("No current execution active")
        
        execution_node = self.graph.get_node(self.current_execution_id)
        if execution_node:
            execution_node.update_property(f"{PROV_NAMESPACE}endedAtTime", 
                                         datetime.now().isoformat())
            execution_node.update_property(f"{KGBENCH_NAMESPACE}hasStatus", 
                                         get_execution_status_iri(status))
            execution_node.update_property(f"{KGBENCH_NAMESPACE}hasExitCode", 
                                         exit_code)
            if logs:
                execution_node.update_property(f"{KGBENCH_NAMESPACE}hasLogs", logs)
        
        self.current_execution_id = None


class DataLineageTracker:
    """Tracks data lineage and transformations."""
    
    def __init__(self, graph: ProvenanceGraph):
        self.graph = graph
    
    def add_data_node(self, data_id: str, filepath: str, 
                     data_format: str, size: Optional[int] = None) -> None:
        """Add a data node to the graph."""
        # Calculate checksum if file exists
        checksum = None
        if Path(filepath).exists():
            with open(filepath, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
        
        data_node = ProvenanceNode(
            iri=data_id,
            node_type=f"{KGBENCH_NAMESPACE}Data",
            properties={
                f"{KGBENCH_NAMESPACE}hasPath": filepath,
                f"{KGBENCH_NAMESPACE}hasFormat": get_data_format_iri(data_format),
                f"{PROV_NAMESPACE}generatedAtTime": datetime.now().isoformat()
            }
        )
        
        if size is not None:
            data_node.update_property(f"{KGBENCH_NAMESPACE}hasSize", size)
        if checksum:
            data_node.update_property(f"{KGBENCH_NAMESPACE}hasChecksum", checksum)
        
        self.graph.add_node(data_node)
    
    def add_transformation(self, input_ids: List[str], output_ids: List[str],
                          transformation_id: str) -> None:
        """Add a data transformation relationship."""
        # Link inputs to transformation
        for input_id in input_ids:
            self.graph.add_edge(ProvenanceEdge(
                source_iri=transformation_id,
                target_iri=input_id,
                edge_type=f"{PROV_NAMESPACE}used"
            ))
        
        # Link outputs to transformation
        for output_id in output_ids:
            self.graph.add_edge(ProvenanceEdge(
                source_iri=output_id,
                target_iri=transformation_id,
                edge_type=f"{PROV_NAMESPACE}wasGeneratedBy"
            ))
            
            # Link outputs to inputs (derivation)
            for input_id in input_ids:
                self.graph.add_edge(ProvenanceEdge(
                    source_iri=output_id,
                    target_iri=input_id,
                    edge_type=f"{PROV_NAMESPACE}wasDerivedFrom"
                ))
    
    def get_lineage(self, data_id: str, direction: str = "both") -> List[str]:
        """Get the lineage of a data entity."""
        visited = set()
        lineage = []
        
        def traverse(node_id: str, depth: int = 0):
            if node_id in visited or depth > 10:  # Prevent infinite loops
                return
            visited.add(node_id)
            lineage.append(node_id)
            
            neighbors = self.graph.get_neighbors(node_id, direction)
            for neighbor in neighbors:
                traverse(neighbor, depth + 1)
        
        traverse(data_id)
        return lineage


class RegistryExporter:
    """Exports registry information to the provenance graph."""
    
    def __init__(self, graph: ProvenanceGraph):
        self.graph = graph
    
    def export_task_registry(self, registry) -> None:
        """Export task registry to the graph."""
        for task_name, task_info in registry._tasks.items():
            task_iri = f"{KGBENCH_NAMESPACE}Task_{task_name}"
            
            # Create task node
            task_node = ProvenanceNode(
                iri=task_iri,
                node_type=f"{KGBENCH_NAMESPACE}Task",
                properties={
                    f"{RDFS_NAMESPACE}label": task_name,
                    f"{KGBENCH_NAMESPACE}hasImplementation": task_info.get('implementation', ''),
                    f"{KGBENCH_NAMESPACE}hasVersion": task_info.get('version', '1.0.0')
                }
            )
            
            # Add category if available
            if 'category' in task_info:
                task_node.update_property(
                    f"{KGBENCH_NAMESPACE}hasCategory",
                    get_task_category_iri(task_info['category'])
                )
            
            # Add Docker image if available
            if 'docker_image' in task_info:
                task_node.update_property(
                    f"{KGBENCH_NAMESPACE}hasDockerImage",
                    task_info['docker_image']
                )
            
            self.graph.add_node(task_node)
    
    def export_pipeline_registry(self, registry) -> None:
        """Export pipeline registry to the graph."""
        for pipeline_name, pipeline_info in registry._pipelines.items():
            pipeline_iri = f"{KGBENCH_NAMESPACE}Pipeline_{pipeline_name}"
            
            # Create pipeline node
            pipeline_node = ProvenanceNode(
                iri=pipeline_iri,
                node_type=f"{KGBENCH_NAMESPACE}Pipeline",
                properties={
                    f"{RDFS_NAMESPACE}label": pipeline_name,
                    f"{KGBENCH_NAMESPACE}hasVersion": pipeline_info.get('version', '1.0.0')
                }
            )
            
            self.graph.add_node(pipeline_node)
            
            # Link to tasks
            for task_name in pipeline_info.get('tasks', []):
                task_iri = f"{KGBENCH_NAMESPACE}Task_{task_name}"
                self.graph.add_edge(ProvenanceEdge(
                    source_iri=pipeline_iri,
                    target_iri=task_iri,
                    edge_type=f"{KGBENCH_NAMESPACE}hasTask"
                ))
    
    def export_format_registry(self, registry) -> None:
        """Export format registry to the graph."""
        for format_name, format_info in registry._formats.items():
            format_iri = f"{KGBENCH_NAMESPACE}Format_{format_name}"
            
            format_node = ProvenanceNode(
                iri=format_iri,
                node_type=f"{KGBENCH_NAMESPACE}DataFormat",
                properties={
                    f"{RDFS_NAMESPACE}label": format_name,
                    f"{RDFS_NAMESPACE}comment": format_info.get('description', '')
                }
            )
            
            self.graph.add_node(format_node) 