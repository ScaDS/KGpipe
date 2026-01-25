# from re import sub
# # from kgcore.common.types import new_id, KGId, Lit, Props
# # from kgcore.api import KGGraph
# # from kgpipe.meta.systemgraph import SYS_KG
# from rdflib import Graph, URIRef, RDF, Literal, BNode
# from typing import Any, Mapping, List, Dict, Tuple, Optional
# from pydantic import BaseModel

# # You can adjust these models if your KGGraph expects different field names,
# # but this shape matches the usage in insert_kg_obj().
# class PreparedNode(BaseModel):
#     id: KGId
#     types: List[str] = []
#     props: Props = {}

# class PreparedEdge(BaseModel):
#     name: str           # IRI of the predicate as string
#     source: KGId        # subject node id
#     target: KGId        # object node id

# def _literal_to_python(value: Literal) -> Lit:
#     """
#     Convert an rdflib Literal to a plain Python value (while preserving language/typed values).
#     If you need richer handling (e.g., keep datatype/lang), adapt here.
#     """
#     try:
#         return value.toPython()
#     except Exception:
#         # Fallback to lexical form as string
#         return str(value)

# def create_insertable_nodes_and_edges(obj) -> Tuple[List[PreparedNode], List[PreparedEdge], URIRef]:
#     """
#     Use a pydantic BaseModel or dict (or any supported object) to construct graph entities
#     for the KGGraph API. Assumes an `object_to_graph(obj)` function exists and returns
#     an rdflib.Graph of triples.

#     Rules:
#       - For (s, rdf:type, o) with o a URIRef, append str(o) to node.types for subject s.
#       - For (s, p, o) with o a Literal, set node.props[str(p)] = value. If multiple values
#         appear for the same (s, p), store them as a list.
#       - For (s, p, o) with o a URIRef, create an edge from s to o, name=str(p).
#       - Subjects/objects that are URIRefs become nodes. Blank nodes are skipped by default.
#     """
#     subject, graph = object_to_graph(obj)

#     # Internal indices
#     iri_to_node: Dict[URIRef, PreparedNode] = {}
#     edges: List[PreparedEdge] = []

#     # To avoid duplicate edges when the same triple repeats
#     edge_seen: set[Tuple[str, KGId, KGId]] = set()

#     def ensure_node(u: URIRef) -> PreparedNode:
#         if u not in iri_to_node:
#             # TODO types
#             iri_to_node[u] = PreparedNode(id=str(u), types=["Some"], props={})
#         return iri_to_node[u]

#     for s, p, o in graph.triples((None, None, None)):
#         # We only support subjects that can map to a concrete node (URI).
#         if not isinstance(s, URIRef):
#             # Optionally, you could map BNodes to generated nodes by treating them similarly:
#             # if isinstance(s, BNode): handle here
#             continue

#         # Handle rdf:type assertions
#         if p == RDF.type and isinstance(o, URIRef):
#             node = ensure_node(s)
#             t = str(o)
#             if t not in node.types:
#                 node.types.append(t)
#             continue

#         # Literal props
#         if isinstance(o, Literal):
#             node = ensure_node(s)
#             key = str(p)
#             val = _literal_to_python(o)

#             if key not in node.props:
#                 node.props[key] = val
#             else:
#                 # Normalize to list for multi-valued properties
#                 existing = node.props[key]
#                 if isinstance(existing, list):
#                     existing.append(val)
#                 else:
#                     node.props[key] = [existing, val]
#             continue

#         # URIRef object => edge; ensure both nodes exist
#         if isinstance(o, URIRef):
#             src = ensure_node(s)
#             tgt = ensure_node(o)
#             name = str(p)
#             sig = (name, src.id, tgt.id)
#             if sig not in edge_seen:
#                 edges.append(PreparedEdge(name=name, source=src.id, target=tgt.id))
#                 edge_seen.add(sig)
#             continue

#         # Skip other node types (e.g., BNodes as objects) by default
#         if isinstance(o, BNode):
#             # If you prefer to materialize BNodes into nodes, you could generate IDs here
#             # and attach their literal properties via a second pass.
#             continue

#     nodes = list(iri_to_node.values())
#     return nodes, edges, subject

# def insert_kg_obj(obj) -> KGId: 
#     entities, relations, subject = create_insertable_nodes_and_edges(obj) 
#     for ent in entities: 
#         SYS_KG.create_entity(labels=ent.types, props=ent.props, id=ent.id) 
#     for rel in relations: 
#         SYS_KG.create_relation(rel.name, rel.source, rel.target)

#     return subject
# # ---

# def object_to_graph(
#     obj: Any,
#     *,
#     subject_ns: str = "urn:res:",
#     predicate_ns: str = "urn:prop:",
#     class_ns: str = "urn:cls:",
#     type_predicate: URIRef = RDF.type,
# ) -> Tuple[URIRef, Graph]:
#     def to_predicate_uri(name: str) -> URIRef:
#         if isinstance(name, str) and (name.startswith("http://") or name.startswith("https://") or name.startswith("urn:")):
#             return URIRef(name)
#         return URIRef(predicate_ns + str(name))

#     def new_subject() -> URIRef:
#         return URIRef(subject_ns + str(new_id()))

#     # --- CHANGE 1: dump with aliases so Field(alias='@type') is preserved ---
#     def to_mapping(value: Any) -> Mapping[str, Any] | None:
#         if isinstance(value, BaseModel):
#             # by_alias=True lets a field alias like '@type' show up
#             return value.model_dump(by_alias=True, exclude_none=True)
#         if isinstance(value, Mapping):
#             return value
#         return None

#     def is_iterable(value: Any) -> bool:
#         return isinstance(value, (list, tuple, set))

#     # Helper: absolute IRI?
#     def _is_abs_iri(s: Any) -> bool:
#         return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("urn:"))

 

#     def class_iri_for(value: Any, mapping: Mapping[str, Any] | None) -> URIRef | None:

#         # 1) instance or class-level __rdf_type__
#         iri = getattr(value, "__rdf_type__", None) or getattr(type(value), "__rdf_type__", None)
#         if _is_abs_iri(iri):
#             return URIRef(iri)

#         # 2) model_config on instance or class
#         cfg = getattr(value, "model_config", None) or getattr(type(value), "model_config", None) or {}
#         iri = (cfg or {}).get("rdf_type")
#         if _is_abs_iri(iri):
#             return URIRef(iri)

#         # 3) enhanced dict-type detection (works for dumped models if you aliased a field)
#         if mapping:
#             iri = mapping.get("@type") or mapping.get("type") or mapping.get("rdf_type") or mapping.get("__rdf_type__")
#             if iri:
#                 return URIRef(iri)

#         # 4) fallback for BaseModel -> urn:cls:ClassName
#         if isinstance(value, BaseModel):
#             return URIRef(class_ns + value.__class__.__name__)

#         return None

#     g = Graph()
#     root_map = to_mapping(obj)
#     if root_map is None:
#         raise TypeError("add_object expects a Pydantic BaseModel or a dict-like object.")

#     root_subject = new_subject()

#     root_type = class_iri_for(obj, root_map)
#     if root_type is not None:
#         g.add((root_subject, type_predicate, root_type))

#     def add_value(head: URIRef, key: str, value: Any) -> None:
#         if value is None:
#             return

#         nested_map = to_mapping(value)
#         if nested_map is not None:
#             tail = new_subject()
#             g.add((head, to_predicate_uri(key), tail))
#             t = class_iri_for(value, nested_map)
#             if t is not None:
#                 g.add((tail, type_predicate, t))
#             for k, v in nested_map.items():
#                 if k in ("@type", "type", "rdf_type"):  # already emitted
#                     continue
#                 add_value(tail, k, v)
#             return

#         if is_iterable(value):
#             for item in value:
#                 add_value(head, key, item)
#             return

#         lit = Literal(value) if isinstance(value, (str, int, float, bool)) else Literal(str(value))
#         g.add((head, to_predicate_uri(key), lit))

#     for k, v in root_map.items():
#         if k in ("@type", "type", "rdf_type"):
#             continue
#         add_value(root_subject, k, v)

#     return root_subject, g


# # --- Example Pydantic models ---
# class Person(BaseModel):
#     id: str
#     name: str
#     age: int
#     friend: Optional["Person"] = None  # forward ref for nesting


# def test_create_insertable_nodes_and_edges():
#     """
#     Test that nested BaseModel objects produce nodes and edges correctly.
#     """

#     # Create two nested people (Alice knows Bob)
#     bob = Person(id="http://example.org/bob", name="Bob", age=25)
#     alice = Person(id="http://example.org/alice", name="Alice", age=30, friend=bob)

#     # Call function under test
#     nodes, edges = create_insertable_nodes_and_edges(alice)

#     print(nodes)
#     print(edges)

#     # --- Assertions ---
#     # We expect two nodes (Alice and Bob)
#     assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"

#     # Each node should have a valid KGId and contain literal props
#     for node in nodes:
#         assert isinstance(node.id, KGId)
#         assert isinstance(node.props, dict)
#         assert "name" in node.props or any("name" in k for k in node.props.keys())

#     # Alice should have the name Alice and Bob should have name Bob
#     names = [n.props.get("name") or list(n.props.values())[0] for n in nodes]
#     # assert "Alice" in str(names) and "Bob" in str(names)

#     # Expect exactly one edge: Alice → friend → Bob
#     assert len(edges) == 1, f"Expected 1 edge, got {len(edges)}"
#     edge = edges[0]
#     assert isinstance(edge, PreparedEdge)
#     assert "friend" in edge.name or "knows" in edge.name.lower()

#     # Edge connects two known nodes
#     node_ids = [n.id for n in nodes]
#     assert edge.source in node_ids
#     assert edge.target in node_ids

#     print("✅ Nodes:")
#     for n in nodes:
#         print("  ", n)

#     print("✅ Edges:")
#     for e in edges:
#         print("  ", e)


# if __name__ == "__main__":
#     test_create_insertable_nodes_and_edges()

def encode_string(s: str) -> str:
    return s.replace(" ", "_").lower()