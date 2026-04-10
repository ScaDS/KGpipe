from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskCategory:
    name: str
    parent: Optional[TaskCategory] = None
    description: str = ""





class BasicTaskCategoryCatalog:
    """
    Hierarchical catalog for task categories.
    Supports default categories and custom category registration.
    """
    entity_resolution = TaskCategory(name="EntityResolution")
    entity_matching = TaskCategory(name="EntityMatching", parent=entity_resolution)
    fusion = TaskCategory(name="Fusion", parent=entity_resolution)
    information_extraction = TaskCategory(name="InformationExtraction")
    entity_linking = TaskCategory(name="EntityLinking", parent=information_extraction)
    relation_extraction = TaskCategory(name="RelationExtraction", parent=information_extraction)
    relation_linking = TaskCategory(name="RelationLinking", parent=information_extraction)
    data_mapping = TaskCategory(name="DataMapping")
    blocking = TaskCategory(name="Blocking", parent=entity_resolution)
    clustering = TaskCategory(name="Clustering", parent=entity_resolution)


    # @dataclass(frozen=True)
    # class TaskCategoryNode:
    #     name: str
    #     parent: Optional[str] = None
    #     description: str = ""

    # _nodes: Dict[str, TaskCategoryNode] = {
    #     "TaskCategory": TaskCategoryNode(name="TaskCategory", parent=None, description="Root category"),
    #     "EntityResolution": TaskCategoryNode(name="EntityResolution", parent="TaskCategory"),
    #     "Blocking": TaskCategoryNode(name="Blocking", parent="EntityResolution"),
    #     "EntityMatching": TaskCategoryNode(name="EntityMatching", parent="EntityResolution"),
    #     "Matching": TaskCategoryNode(name="Matching", parent="EntityResolution"),
    #     "Clustering": TaskCategoryNode(name="Clustering", parent="EntityResolution"),
    #     "Fusion": TaskCategoryNode(name="Fusion", parent="EntityResolution"),
    #     "InformationExtraction": TaskCategoryNode(name="InformationExtraction", parent="TaskCategory"),
    #     "EntityLinking": TaskCategoryNode(name="EntityLinking", parent="InformationExtraction"),
    #     "RelationExtraction": TaskCategoryNode(name="RelationExtraction", parent="InformationExtraction"),
    #     "RelationLinking": TaskCategoryNode(name="RelationLinking", parent="InformationExtraction"),
    #     "DataMapping": TaskCategoryNode(name="DataMapping", parent="TaskCategory"),
    # }

    # @classmethod
    # def has(cls, category: str) -> bool:
    #     return category in cls._nodes

    # @classmethod
    # def register(cls, name: str, parent: str = "TaskCategory", description: str = "") -> None:
    #     if parent is not None and parent not in cls._nodes:
    #         raise ValueError(f"Unknown parent category: {parent}")
    #     cls._nodes[name] = TaskCategoryNode(name=name, parent=parent, description=description)

    # @classmethod
    # def get_parent(cls, category: str) -> Optional[str]:
    #     node = cls._nodes.get(category)
    #     if node is None:
    #         raise ValueError(f"Unknown category: {category}")
    #     return node.parent

    # @classmethod
    # def get_children(cls, category: str) -> List[str]:
    #     if category not in cls._nodes:
    #         raise ValueError(f"Unknown category: {category}")
    #     return sorted([node.name for node in cls._nodes.values() if node.parent == category])

    # @classmethod
    # def get_ancestors(cls, category: str) -> List[str]:
    #     if category not in cls._nodes:
    #         raise ValueError(f"Unknown category: {category}")
    #     ancestors: List[str] = []
    #     cursor = cls._nodes[category].parent
    #     while cursor is not None:
    #         ancestors.append(cursor)
    #         cursor = cls._nodes[cursor].parent
    #     return ancestors

    # @classmethod
    # def get_descendants(cls, category: str) -> List[str]:
    #     if category not in cls._nodes:
    #         raise ValueError(f"Unknown category: {category}")
    #     descendants: List[str] = []
    #     queue = cls.get_children(category)
    #     while queue:
    #         current = queue.pop(0)
    #         descendants.append(current)
    #         queue.extend(cls.get_children(current))
    #     return descendants

    # @classmethod
    # def is_subtask_of(cls, category: str, parent: str) -> bool:
    #     if category not in cls._nodes or parent not in cls._nodes:
    #         return False
    #     return parent in cls.get_ancestors(category)

    # @classmethod
    # def list_categories(cls) -> List[str]:
    #     return sorted(cls._nodes.keys())


class BasicDataFormats(str, Enum):
    """Framework-provided data formats with IDE autocomplete."""

    # Standard formats
    RDF_TTL = "ttl"
    RDF_NQUADS = "nq"
    RDF_NTRIPLES = "nt"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    RDF_XML = "xml"
    RDF = "rdf"
    RDF_JSONLD = "jsonld"
    TEXT = "txt"
    XML = "xml"
    ANY = "any"

    # Tool-specific formats
    PARIS_CSV = "paris.csv"
    OPENREFINE_JSON = "openrefine.json"
    LIMES_XML = "limes.xml"
    SPOTLIGHT_JSON = "spotlight.json"
    FALCON_JSON = "falcon.json"
    VALENTINE_JSON = "valentine.json"
    CORENLP_JSON = "corenlp.json"
    OPENIE_JSON = "openie.json"
    AGREEMENTMAKER_RDF = "agreementmaker.rdf"

    # Exchange formats
    ER_JSON = "er.json"
    TE_JSON = "te.json"

    # LLM task outputs
    JSON_ONTO_MAPPING_JSON = "json_onto_mapping.json"

    @property
    def extension(self) -> str:
        return self.value

    @property
    def description(self) -> str:
        return BASIC_FORMAT_DESCRIPTIONS.get(self.value, self.value)

    @property
    def is_tool_specific(self) -> bool:
        return "." in self.value and self.value not in {"jsonld"}

    @classmethod
    def from_extension(cls, extension: str) -> "BasicDataFormats":
        try:
            return cls(extension)
        except ValueError as exc:
            available = [f.value for f in cls]
            raise ValueError(f"Invalid format: {extension}. Available formats: {available}") from exc


class CustomDataFormats(str, Enum):
    """
    Base enum for user-defined formats.
    Define project-specific formats by subclassing this enum.
    """

    @property
    def extension(self) -> str:
        return self.value


BASIC_FORMAT_DESCRIPTIONS: dict[str, str] = {
    "ttl": "Turtle RDF format",
    "nq": "N-Quads RDF format",
    "json": "JSON format",
    "csv": "CSV format",
    "parquet": "Parquet format",
    "xml": "XML format",
    "rdf": "RDF format",
    "jsonld": "JSON-LD format",
    "txt": "Text format",
    "paris.csv": "Paris CSV format",
    "openrefine.json": "OpenRefine JSON format",
    "limes.xml": "LIMES XML format",
    "spotlight.json": "DBpedia Spotlight JSON format",
    "falcon.json": "FALCON JSON format",
    "valentine.json": "Valentine JSON format",
    "corenlp.json": "CoreNLP JSON format",
    "openie.json": "OpenIE JSON format",
    "agreementmaker.rdf": "AgreementMaker RDF format",
    "er.json": "Entity Resolution JSON format",
    "te.json": "Text Extraction JSON format",
    "json_onto_mapping.json": "JSON ontology mapping format",
    "any": "Any format",
}