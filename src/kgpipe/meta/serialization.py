"""
Serialization for KGbench Meta Knowledge Graph

This module provides serialization formats for the meta knowledge graph,
including RDF, JSON-LD, and GraphQL representations.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from abc import ABC, abstractmethod

from .provenance import ProvenanceGraph


class ProvenanceSerializer(ABC):
    """Abstract base class for provenance serializers."""
    
    @abstractmethod
    def serialize(self, graph: ProvenanceGraph) -> str:
        """Serialize the graph to a string representation."""
        pass
    
    @abstractmethod
    def save(self, graph: ProvenanceGraph, filepath: Union[str, Path]) -> None:
        """Save the graph to a file."""
        pass


class RDFSerializer(ProvenanceSerializer):
    """Serializer for RDF/Turtle format."""
    
    def __init__(self):
        self.namespaces = {
            "kgpipe": "http://kgpipe.org/ontology#",
            "prov": "http://www.w3.org/ns/prov#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "dcterms": "http://purl.org/dc/terms/"
        }
    
    def serialize(self, graph: ProvenanceGraph) -> str:
        """Serialize the graph to RDF/Turtle format."""
        lines = []
        
        # Add namespace declarations
        for prefix, uri in self.namespaces.items():
            lines.append(f"@prefix {prefix}: <{uri}> .")
        lines.append("")
        
        # Add graph metadata
        lines.append(f"# KGbench Provenance Graph: {graph.name}")
        lines.append(f"# Created: {graph.created_at.isoformat()}")
        lines.append("")
        
        # Serialize nodes
        for iri, node in graph.nodes.items():
            lines.append(f"# Node: {node.node_type}")
            lines.append(f"<{iri}> a <{node.node_type}> ;")
            
            # Add properties
            prop_lines = []
            for key, value in node.properties.items():
                if isinstance(value, str):
                    if value.startswith("http"):
                        prop_lines.append(f"    <{key}> <{value}>")
                    else:
                        prop_lines.append(f'    <{key}> "{value}"')
                elif isinstance(value, (int, float)):
                    prop_lines.append(f"    <{key}> {value}")
                elif isinstance(value, bool):
                    prop_lines.append(f"    <{key}> {str(value).lower()}")
                else:
                    prop_lines.append(f'    <{key}> "{str(value)}"')
            
            if prop_lines:
                lines.extend(prop_lines)
                lines.append("    .")
            else:
                lines.append("    .")
            lines.append("")
        
        # Serialize edges
        for edge in graph.edges:
            lines.append(f"# Edge: {edge.edge_type}")
            lines.append(f"<{edge.source_iri}> <{edge.edge_type}> <{edge.target_iri}> .")
            lines.append("")
        
        return "\n".join(lines)
    
    def save(self, graph: ProvenanceGraph, filepath: Union[str, Path]) -> None:
        """Save the graph to a Turtle file."""
        content = self.serialize(graph)
        with open(filepath, 'w') as f:
            f.write(content)


class JSONLDSerializer(ProvenanceSerializer):
    """Serializer for JSON-LD format."""
    
    def __init__(self):
        self.context = {
            "@context": {
                "kgpipe": "http://kgpipe.org/ontology#",
                "prov": "http://www.w3.org/ns/prov#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "dcterms": "http://purl.org/dc/terms/",
                "type": "@type",
                "id": "@id"
            }
        }
    
    def serialize(self, graph: ProvenanceGraph) -> str:
        """Serialize the graph to JSON-LD format."""
        graph_data: Dict[str, Any] = {
            "@context": self.context["@context"],
            "@graph": []
        }
        
        # Add nodes
        for iri, node in graph.nodes.items():
            node_data: Dict[str, Any] = {
                "@id": iri,
                "@type": node.node_type
            }
            
            # Add properties
            for key, value in node.properties.items():
                if isinstance(value, str) and value.startswith("http"):
                    node_data[str(key)] = {"@id": value}
                else:
                    node_data[str(key)] = value
            
            graph_data["@graph"].append(node_data)
        
        # Add edges as additional nodes
        for edge in graph.edges:
            edge_data: Dict[str, Any] = {
                "@id": f"{edge.source_iri}_{edge.edge_type}_{edge.target_iri}",
                "@type": edge.edge_type,
                "prov:source": {"@id": edge.source_iri},
                "prov:target": {"@id": edge.target_iri}
            }
            
            # Add edge properties
            for key, value in edge.properties.items():
                edge_data[str(key)] = value
            
            graph_data["@graph"].append(edge_data)
        
        return json.dumps(graph_data, indent=2)
    
    def save(self, graph: ProvenanceGraph, filepath: Union[str, Path]) -> None:
        """Save the graph to a JSON-LD file."""
        content = self.serialize(graph)
        with open(filepath, 'w') as f:
            f.write(content)


class GraphQLSerializer(ProvenanceSerializer):
    """Serializer for GraphQL schema and queries."""
    
    def __init__(self):
        self.schema = """
type ProvenanceGraph {
    name: String!
    createdAt: String!
    nodes: [Node!]!
    edges: [Edge!]!
}

type Node {
    iri: String!
    type: String!
    properties: [Property!]!
    createdAt: String!
    updatedAt: String!
}

type Property {
    key: String!
    value: String!
}

type Edge {
    source: String!
    target: String!
    type: String!
    properties: [Property!]!
    createdAt: String!
}

type Query {
    graph(name: String!): ProvenanceGraph
    node(iri: String!): Node
    edges(source: String, target: String, type: String): [Edge!]!
    neighbors(iri: String!, direction: String): [String!]!
    lineage(dataId: String!, direction: String): [String!]!
}
"""
    
    def serialize(self, graph: ProvenanceGraph) -> str:
        """Serialize the graph to GraphQL schema and sample queries."""
        lines = []
        lines.append("# GraphQL Schema for KGbench Provenance")
        lines.append(self.schema)
        lines.append("")
        
        # Add sample queries
        lines.append("# Sample Queries")
        lines.append("")
        
        # Query for graph
        lines.append("# Get entire graph")
        lines.append("query GetGraph($name: String!) {")
        lines.append("  graph(name: $name) {")
        lines.append("    name")
        lines.append("    createdAt")
        lines.append("    nodes {")
        lines.append("      iri")
        lines.append("      type")
        lines.append("      properties {")
        lines.append("        key")
        lines.append("        value")
        lines.append("      }")
        lines.append("    }")
        lines.append("    edges {")
        lines.append("      source")
        lines.append("      target")
        lines.append("      type")
        lines.append("    }")
        lines.append("  }")
        lines.append("}")
        lines.append("")
        
        # Query for specific node
        lines.append("# Get specific node")
        lines.append("query GetNode($iri: String!) {")
        lines.append("  node(iri: $iri) {")
        lines.append("    iri")
        lines.append("    type")
        lines.append("    properties {")
        lines.append("      key")
        lines.append("      value")
        lines.append("    }")
        lines.append("  }")
        lines.append("}")
        lines.append("")
        
        # Query for edges
        lines.append("# Get edges")
        lines.append("query GetEdges($source: String, $target: String, $type: String) {")
        lines.append("  edges(source: $source, target: $target, type: $type) {")
        lines.append("    source")
        lines.append("    target")
        lines.append("    type")
        lines.append("  }")
        lines.append("}")
        lines.append("")
        
        # Query for neighbors
        lines.append("# Get neighbors")
        lines.append("query GetNeighbors($iri: String!, $direction: String) {")
        lines.append("  neighbors(iri: $iri, direction: $direction)")
        lines.append("}")
        lines.append("")
        
        # Query for lineage
        lines.append("# Get data lineage")
        lines.append("query GetLineage($dataId: String!, $direction: String) {")
        lines.append("  lineage(dataId: $dataId, direction: $direction)")
        lines.append("}")
        lines.append("")
        
        # Add sample data
        lines.append("# Sample Variables")
        lines.append("""{
  "name": "kgpipe_provenance",
  "iri": "http://kgpipe.org/ontology#Task_paris_matcher",
  "source": "http://kgpipe.org/ontology#Task_paris_matcher",
  "target": "http://kgpipe.org/ontology#Data_output_1",
  "type": "http://www.w3.org/ns/prov#wasGeneratedBy",
  "dataId": "http://kgpipe.org/ontology#Data_output_1",
  "direction": "both"
}""")
        
        return "\n".join(lines)
    
    def save(self, graph: ProvenanceGraph, filepath: Union[str, Path]) -> None:
        """Save the GraphQL schema and queries to a file."""
        content = self.serialize(graph)
        with open(filepath, 'w') as f:
            f.write(content)


class SPARQLSerializer(ProvenanceSerializer):
    """Serializer for SPARQL queries."""
    
    def serialize(self, graph: ProvenanceGraph) -> str:
        """Serialize the graph to SPARQL queries."""
        lines = []
        lines.append("# SPARQL Queries for KGbench Provenance")
        lines.append("")
        
        # PREFIX declarations
        lines.append("PREFIX kgpipe: <http://kgpipe.org/ontology#>")
        lines.append("PREFIX prov: <http://www.w3.org/ns/prov#>")
        lines.append("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
        lines.append("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>")
        lines.append("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>")
        lines.append("PREFIX dcterms: <http://purl.org/dc/terms/>")
        lines.append("")
        
        # Query 1: Get all tasks
        lines.append("# Get all tasks")
        lines.append("SELECT ?task ?label ?category ?implementation")
        lines.append("WHERE {")
        lines.append("  ?task a kgpipe:Task .")
        lines.append("  OPTIONAL { ?task rdfs:label ?label }")
        lines.append("  OPTIONAL { ?task kgpipe:hasCategory ?category }")
        lines.append("  OPTIONAL { ?task kgpipe:hasImplementation ?implementation }")
        lines.append("}")
        lines.append("")
        
        # Query 2: Get all executions
        lines.append("# Get all executions")
        lines.append("SELECT ?execution ?status ?startTime ?endTime")
        lines.append("WHERE {")
        lines.append("  ?execution a kgpipe:Execution .")
        lines.append("  OPTIONAL { ?execution kgpipe:hasStatus ?status }")
        lines.append("  OPTIONAL { ?execution prov:startedAtTime ?startTime }")
        lines.append("  OPTIONAL { ?execution prov:endedAtTime ?endTime }")
        lines.append("}")
        lines.append("")
        
        # Query 3: Get data lineage
        lines.append("# Get data lineage")
        lines.append("SELECT ?data ?derivedFrom ?generatedBy")
        lines.append("WHERE {")
        lines.append("  ?data a kgpipe:Data .")
        lines.append("  OPTIONAL { ?data prov:wasDerivedFrom ?derivedFrom }")
        lines.append("  OPTIONAL { ?data prov:wasGeneratedBy ?generatedBy }")
        lines.append("}")
        lines.append("")
        
        # Query 4: Get execution workflow
        lines.append("# Get execution workflow")
        lines.append("SELECT ?execution ?task ?input ?output")
        lines.append("WHERE {")
        lines.append("  ?execution a kgpipe:Execution .")
        lines.append("  ?task prov:wasAssociatedWith ?execution .")
        lines.append("  OPTIONAL { ?task prov:used ?input }")
        lines.append("  OPTIONAL { ?output prov:wasGeneratedBy ?task }")
        lines.append("}")
        lines.append("")
        
        # Query 5: Get metrics
        lines.append("# Get metrics")
        lines.append("SELECT ?metric ?type ?value ?unit")
        lines.append("WHERE {")
        lines.append("  ?metric a kgpipe:Metric .")
        lines.append("  OPTIONAL { ?metric kgpipe:hasType ?type }")
        lines.append("  OPTIONAL { ?metric kgpipe:hasValue ?value }")
        lines.append("  OPTIONAL { ?metric kgpipe:hasUnit ?unit }")
        lines.append("}")
        lines.append("")
        
        # Query 6: Get task performance
        lines.append("# Get task performance")
        lines.append("SELECT ?task ?executionTime ?memoryUsage ?cpuUsage")
        lines.append("WHERE {")
        lines.append("  ?task a kgpipe:Task .")
        lines.append("  ?execution prov:wasAssociatedWith ?task .")
        lines.append("  ?metric prov:wasAttributedTo ?execution .")
        lines.append("  ?metric kgpipe:hasType kgpipe:EXECUTION_TIME .")
        lines.append("  ?metric kgpipe:hasValue ?executionTime .")
        lines.append("  OPTIONAL {")
        lines.append("    ?memoryMetric kgpipe:hasType kgpipe:MEMORY_USAGE .")
        lines.append("    ?memoryMetric kgpipe:hasValue ?memoryUsage .")
        lines.append("  }")
        lines.append("  OPTIONAL {")
        lines.append("    ?cpuMetric kgpipe:hasType kgpipe:CPU_USAGE .")
        lines.append("    ?cpuMetric kgpipe:hasValue ?cpuUsage .")
        lines.append("  }")
        lines.append("}")
        
        return "\n".join(lines)
    
    def save(self, graph: ProvenanceGraph, filepath: Union[str, Path]) -> None:
        """Save the SPARQL queries to a file."""
        content = self.serialize(graph)
        with open(filepath, 'w') as f:
            f.write(content) 