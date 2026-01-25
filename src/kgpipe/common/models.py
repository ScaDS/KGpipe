"""
KGflex model.py
Core domain model for the KGflex framework.

This module defines the core data structures that represent the domain model
for generating and executing KG pipelines.
"""

from __future__ import annotations

from .model.data import Data, DataFormat, DynamicFormat, DataSet, FormatRegistry
from .model.task import KgTask, KgTaskReport
from .model.pipeline import KgPipe, KgPipePlan, KgPipePlanStep
from .model.evaluation import Metric, EvaluationReport
from .model.kg import KG
from .model.task import TaskInput, TaskOutput

__all__ = [
    "Data", "DataFormat", "DynamicFormat", "DataSet", "FormatRegistry", "KgTask", "KgTaskReport", "KgPipe", "KgPipePlan", "KgPipePlanStep",  "Metric", "EvaluationReport", "KG", "TaskInput", "TaskOutput"
]

# TODO remove this for next release
# @dataclass
# class KG:
#     """Represents a knowledge graph."""
#     id: str
#     name: str
#     path: Path
#     format: Format
#     triple_count: Optional[int] = None
#     entity_count: Optional[int] = None
#     description: Optional[str] = None
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     graph: Optional[Graph] = None
#     data_graph: Optional[Graph] = None
#     ontology_graph: Optional[Graph] = None
#     plan: Optional[KgPipePlan] = None

#     def __post_init__(self):
#         if not self.id:
#             self.id = str(uuid.uuid4())
#         if isinstance(self.path, str):
#             self.path = Path(self.path)
#         if not self.name:
#             raise ValueError("KG name cannot be empty")

#     def get_graph(self) -> Graph:
#         if self.graph is None:
#             tmp = Graph().parse(self.path)
#             graph = Graph()
#             for s, p, o in tmp:
#                 if (str(p) != str(SKOS.altLabel)):
#                     graph.add((s, p, o))
#             self.graph = graph
#         return self.graph

#     def get_data_graph(self) -> Graph:
#         return Graph()
    
#     def get_ontology_graph(self) -> Graph:
#         # TODO derive from graph
#         if self.ontology_graph is None:
#             self.ontology_graph = Graph()
#         return self.ontology_graph

#     def set_ontology_graph(self, graph: Graph) -> None:
#         print(f"Setting ontology graph with {len(graph)} triples")
#         self.ontology_graph = graph

#     def exists(self) -> bool:
#         """Check if the KG file exists."""
#         return self.path.exists()
    
#     def __str__(self) -> str:
#         return f"KG({self.name}, {self.path}, {self.format.value})"





# # Backward compatibility aliases
# Task = KgTask
# Pipeline = KgPipe