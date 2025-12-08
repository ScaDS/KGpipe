# kgcore/backend/rdf_file.py
from __future__ import annotations
from typing import List, Optional
from pathlib import Path
from rdflib import Graph, URIRef, Literal, BNode, RDF
from rdflib.util import guess_format
from kgcore.backend.rdf_rdflib import RDFLibGraph
from kgcore.model.base import KGEntity, KGRelation
from kgcore.common.types import Props, KGId

class RDFFileGraph(RDFLibGraph):
    """RDF backend with file persistence capabilities."""
    
    def __init__(self, file_path: str | Path, base_iri: str = "http://example.org/", 
                 format: str = "turtle"):
        super().__init__(base_iri)
        self.file_path = Path(file_path)
        self.format = format
        self._load_from_file()
    
    def _load_from_file(self) -> None:
        """Load existing RDF data from file if it exists."""
        if self.file_path.exists():
            try:
                self.g.parse(str(self.file_path), format=self.format)
            except Exception as e:
                # If parsing fails, start with empty graph
                print(f"Warning: Could not load RDF file {self.file_path}: {e}")
                self.g = Graph()
    
    def save_to_file(self, file_path: str | Path | None = None, 
                    format: str | None = None) -> None:
        """Save the current graph to file."""
        target_path = Path(file_path) if file_path else self.file_path
        target_format = format or self.format
        
        # Ensure directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize and save
        try:
            self.g.serialize(destination=str(target_path), format=target_format)
        except Exception as e:
            raise RuntimeError(f"Failed to save RDF to {target_path}: {e}")
    
    def create_entity(self, labels: List[str], props: Props | None = None) -> KGEntity:
        """Create entity and auto-save to file."""
        entity = super().create_entity(labels, props)
        self.save_to_file()
        return entity
    
    def create_relation(self, type: str, source: KGId, target: KGId, 
                      props: Props | None = None) -> KGRelation:
        """Create relation and auto-save to file."""
        relation = super().create_relation(type, source, target, props)
        
        # Store relation properties as additional triples
        if props:
            s = URIRef(self.base + source)
            for k, v in props.items():
                self.g.add((s, URIRef(self.base + k), Literal(v)))
        
        self.save_to_file()
        return relation
    
    def add_meta(self, target: KGId, meta: Props) -> None:
        """Add metadata using reification and auto-save to file."""
        super().add_meta(target, meta)
        self.save_to_file()
    
    def find_entities(self, label: str | None = None, props: Props | None = None):
        """Find entities by label and/or properties."""
        if not label and not props:
            return []
        
        entities = []
        processed_subjects = set()
        
        for s, p, o in self.g:
            if p == RDF.type and isinstance(o, URIRef) and s not in processed_subjects:
                type_name = str(o).replace(self.base, "")
                if not label or type_name == label:
                    # Extract entity properties
                    entity_props = {}
                    for s2, p2, o2 in self.g.triples((s, None, None)):
                        if p2 != RDF.type and isinstance(o2, Literal):
                            prop_name = str(p2).replace(self.base, "")
                            entity_props[prop_name] = str(o2)
                    
                    # Check property filters
                    if props:
                        if not all(entity_props.get(k) == str(v) for k, v in props.items()):
                            continue
                    
                    entity = KGEntity(
                        id=str(s).replace(self.base, ""),
                        labels=[type_name],
                        props=entity_props
                    )
                    entities.append(entity)
                    processed_subjects.add(s)
        
        return entities
    
    def find_relations(self, type: str | None = None, props: Props | None = None):
        """Find relations by type and/or properties."""
        if not type and not props:
            return []
        
        relations = []
        processed_triples = set()
        
        for s, p, o in self.g:
            if isinstance(p, URIRef) and p != RDF.type:
                rel_type = str(p).replace(self.base, "")
                if not type or rel_type == type:
                    # Create a unique key for this triple
                    triple_key = (s, p, o)
                    if triple_key not in processed_triples:
                        # Extract relation properties
                        rel_props = {}
                        for s2, p2, o2 in self.g.triples((s, None, None)):
                            if p2 != RDF.type and isinstance(o2, Literal):
                                prop_name = str(p2).replace(self.base, "")
                                rel_props[prop_name] = str(o2)
                        
                        # Check property filters
                        if props:
                            if not all(rel_props.get(k) == str(v) for k, v in props.items()):
                                continue
                        
                        relation = KGRelation(
                            id=str(s).replace(self.base, ""),
                            type=rel_type,
                            source=str(s).replace(self.base, ""),
                            target=str(o).replace(self.base, ""),
                            props=rel_props
                        )
                        relations.append(relation)
                        processed_triples.add(triple_key)
        
        return relations
