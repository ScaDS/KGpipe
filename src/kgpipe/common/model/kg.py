from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from rdflib import Graph, SKOS, RDF

from .data import Format
from .pipeline import KgPipePlan

# TODO check if this is still needed or if we can use the KG from kgcore and only use Data and DataSet

@dataclass
class KG:
    """Represents a knowledge graph."""
    id: str
    name: str
    path: Path
    format: Format
    triple_count: Optional[int] = None
    entity_count: Optional[int] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph: Optional[Graph] = None
    data_graph: Optional[Graph] = None
    ontology_graph: Optional[Graph] = None
    plan: Optional[KgPipePlan] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if not self.name:
            raise ValueError("KG name cannot be empty")

    def get_graph(self) -> Graph:
        if self.graph is None:
            tmp = Graph().parse(self.path)
            graph = Graph()
            for s, p, o in tmp:
                if (str(p) != str(SKOS.altLabel)):
                    graph.add((s, p, o))
            self.graph = graph
        return self.graph

    def get_data_graph(self) -> Graph:
        return Graph()
    
    def get_ontology_graph(self) -> Graph:
        # TODO derive from graph
        if self.ontology_graph is None:
            self.ontology_graph = Graph()
        return self.ontology_graph

    def set_ontology_graph(self, graph: Graph) -> None:
        print(f"Setting ontology graph with {len(graph)} triples")
        self.ontology_graph = graph

    def exists(self) -> bool:
        """Check if the KG file exists."""
        return self.path.exists()
    
    def __str__(self) -> str:
        return f"KG({self.name}, {self.path}, {self.format.value})"


# TODO wip class for central KgPipe KG entity

@dataclass
class KgKg:
    """Represents a KG for the KgPipe framework."""
    graph_data: KgData
    ontology_data: KgData
    # provenance: str
    # @staticmethod
    # def load_from_plan(plan: KgPipePlan) -> KG:
    #     pass
    # pass