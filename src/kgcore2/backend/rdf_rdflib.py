# kgcore/backend/rdf_rdflib.py
from __future__ import annotations
from typing import List
from rdflib import Graph, URIRef, Literal, BNode, RDF
from rdflib.namespace import RDF as RDF_NS
from kgcore.model.base import KGGraph, KGEntity, KGRelation
from kgcore.common.types import Props, KGId

class RDFLibGraph(KGGraph):
    def __init__(self, base_iri: str = "http://example.org/"):
        self.g = Graph()
        self.base = base_iri

    def create_entity(self, labels: List[str], props: Props | None = None, id: KGId = None) -> KGEntity:
        if id:
            e = KGEntity(id=id,labels=list(labels or []), props=dict(props or {}))
        else:
            e = KGEntity(labels=list(labels or []), props=dict(props or {}))

        s = URIRef(self.base + e.id)
        for lbl in labels or []:
            self.g.add((s, RDF.type, URIRef(self.base + lbl)))
        for k, v in (props or {}).items():
            self.g.add((s, URIRef(self.base + k), Literal(v)))
        return e

    def create_relation(self, type: str, source: KGId, target: KGId, props: Props | None = None) -> KGRelation:
        r = KGRelation(type=type, source=source, target=target, props=dict(props or {}))
        s = URIRef(self.base + source); p = URIRef(self.base + type); o = URIRef(self.base + target)
        self.g.add((s, p, o))
        return r

    def add_meta(self, target: KGId, meta: Props) -> None:
        """Add metadata to a target using RDF reification."""
        # Find the triple(s) that match the target ID
        # For entities, target is the subject; for relations, target is the relation ID
        target_uri = URIRef(self.base + target)
        
        # Find all triples where target_uri is the subject
        matching_triples = list(self.g.triples((target_uri, None, None)))
        
        if not matching_triples:
            # If no triples found, the target might be a relation
            # Look for triples where target appears in any position
            for s, p, o in self.g:
                if str(s).replace(self.base, "") == target or str(p).replace(self.base, "") == target or str(o).replace(self.base, "") == target:
                    matching_triples.append((s, p, o))
        
        # Create reification for each matching triple
        for s, p, o in matching_triples:
            # Create a blank node to represent the reified statement
            reification_node = BNode()
            
            # Add the reification triples
            self.g.add((reification_node, RDF_NS.type, RDF_NS.Statement))
            self.g.add((reification_node, RDF_NS.subject, s))
            self.g.add((reification_node, RDF_NS.predicate, p))
            self.g.add((reification_node, RDF_NS.object, o))
            
            # Add the metadata properties to the reification node
            for key, value in meta.items():
                self.g.add((reification_node, URIRef(self.base + key), Literal(value)))

    def find_entities(self, label: str | None = None, props: Props | None = None):
        # Minimal: return empty for now; implement later
        return []

    def find_relations(self, type: str | None = None, props: Props | None = None):
        return []

    def asGraph(self) -> Graph:
        return self.g