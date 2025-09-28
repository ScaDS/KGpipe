from tokenize import Octnumber
from typing import Iterator
import re
import json
import hashlib
import numpy as np
from rdflib import Graph
from rdflib.namespace import SKOS, RDFS
from sentence_transformers import SentenceTransformer
from kgpipe_tasks.text_processing.relation_match import AliasAndTransformerBasedRelationLinker
from kgpipe_tasks.common.ontology import Ontology, OntologyUtil
from kgpipe.common import Registry, DataFormat, Data
from typing import Dict
from pathlib import Path
from tqdm import tqdm

from kgpipe.util.embeddings import global_encode

# --- Config ---
LABEL_KEYS = re.compile(r"(?:^|_)(name|label|title|display.?name|prefLabel)$", re.I)
ID_KEYS    = re.compile(r"(?:^|_)(id|code|sku|gtin|ean|isbn|doi|cas)$", re.I)
THRESHOLD  = 0.70  # tune later

# --- Helpers ---
def find_labelish(obj: dict) -> tuple[str, str] | None:
    """
    return (key, value) if key and value else None
    """
    for k, v in obj.items():
        if isinstance(v, str) and LABEL_KEYS.search(k) and 1 <= len(v) <= 200:
            return k, v.strip()
    return None

def top_level_kv_string(obj: dict) -> str:
    parts = []
    for k, v in obj.items():
        if isinstance(v, (str, int, float)):
            parts.append(f"{k}: {v}")
        elif isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
            parts.append(f"{k}: " + "; ".join(map(str, v)))
    # sort keys for reproducibility
    parts.sort()
    return " | ".join(parts)

def load_kg_labels(ttl_path: str):
    g = Graph(); g.parse(ttl_path)
    items = []  # (uri, text)
    for s, _, o in g.triples((None, SKOS.prefLabel, None)):
        items.append((str(s), str(o)))
    for s, _, o in g.triples((None, SKOS.altLabel, None)):
        items.append((str(s), str(o)))
    for s, _, o in g.triples((None, RDFS.label, None)):
        items.append((str(s), str(o)))
    # de-duplicate by (uri,text)
    seen = set(); dedup = []
    for u,t in items:
        if (u,t) not in seen:
            seen.add((u,t)); dedup.append((u,t))
    return dedup

def normalize(vectors: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / n

# --- Index build ---
class LabelIndex:

    show_progress_bar = False

    def __init__(self, items, encode):
        self.encode = encode
        self.uris  = [u for u,_ in items]
        self.texts = [t for _,t in items]
        # self.vecs  = normalize(encode(self.texts, convert_to_numpy=True, show_progress_bar=self.show_progress_bar))
        self.vecs = normalize(self.encode(self.texts))

    def update_index(self, items: list[tuple[str, str]]):
        self.uris.extend([u for u,_ in items])
        self.texts.extend([t for _,t in items])
        # self.vecs = normalize(self.encode(self.texts, convert_to_numpy=True, show_progress_bar=self.show_progress_bar))
        self.vecs = normalize(self.encode(self.texts))


    def search(self, query_text: str, k: int = 10):
        # q = normalize(self.encode([query_text], convert_to_numpy=True, show_progress_bar=self.show_progress_bar))[0]
        q = normalize(self.encode([query_text]))[0]
        sims = self.vecs @ q  # cosine (since normalized)
        idx = np.argsort(-sims)[:k]
        return [(self.uris[i], self.texts[i], float(sims[i])) for i in idx]

# --- Baseline linker ---
class SimpleEntityLinker:
    def __init__(self, ttl_path: str):
        # self.model = SentenceTransformer(EMB_MODEL)
        items = load_kg_labels(ttl_path)
        self.index = LabelIndex(items, global_encode)

    def update_index(self, new_aliases: list[tuple[str, str]]):
        if new_aliases:
            self.index.update_index(new_aliases)


    def link(self, obj: dict):
        label_kv = find_labelish(obj)
        if label_kv:
            query = label_kv[1]
            strategy = "label-only"
        else:
            query = top_level_kv_string(obj)
            strategy = "top-level-kv"

        candidates = self.index.search(query, k=5)
        best_uri, best_text, best_score = candidates[0]
        decision = "accept" if best_score >= THRESHOLD else "review"

        return {
            "strategy": strategy,
            "query_text": query,
            "best": {"uri": best_uri, "label": best_text, "score": best_score},
            "candidates": candidates,
            "decision": decision
        }


class SimpleRelationLinker:

    def __init__(self, ontology_path: str):
        ontlogy = OntologyUtil.load_ontology_from_file(Path(ontology_path))
        self.properties = ontlogy.properties
        label_uri = []
        for prop in self.properties:
            label_uri.append((prop.uri, prop.label))
            for alias in prop.alias:
                label_uri.append((prop.uri, alias))
        self.label_uri = label_uri
        self.index = LabelIndex(label_uri, global_encode)
        self.cache = {}

    def link(self, obj: dict):
        links = []
        # TODO use literal object heuristic to select a better candidate
        for keys in obj.keys():
            query = keys
            strategy = "key-only"
            if query in self.cache:
                links.append(self.cache[query])
                continue
            
            candidates = self.index.search(query, k=5)
            best_uri, best_text, best_score = candidates[0]
            decision = "accept" if best_score >= THRESHOLD else "review"

            links.append({
                "strategy": strategy,
                "query_text": query,
                "best": {"uri": best_uri, "label": best_text, "score": best_score},
                "candidates": candidates,
                "decision": decision
            })
            self.cache[query] = links[-1]
        return links


# class SimpleJsonToRdfLinker:
#     def __init__(self, kg_path: str, ontology_path: str):
#         self.entity_linker = SimpleEntityLinker(kg_path)
#         self.relation_linker = SimpleRelationLinker(ontology_path)
    
#     def link(self, obj: dict):
        
#         def link_object(obj: dict):
#             res = self.entity_linker.link(obj)
#             links = self.relation_linker.link(obj)
#             return res, links
        
#         for key, value in obj.items():
#             if isinstance(value, dict):
#                 link_object(value)
#             elif isinstance(value, list):
#                 for item in value:
#                     link_object(item)
#             else:
#                 pass

# --- New: JSON→RDF linker ----------------------------------------------------

import re
import uuid
from typing import Dict, List, Tuple, Optional, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from rdflib.namespace import RDF, RDFS, SKOS, XSD, DCTERMS
from sentence_transformers import SentenceTransformer

@dataclass
class LinkDecision:
    node: URIRef | BNode
    created: bool
    label: Optional[str]
    entity_link: Optional[dict] = None  # from SimpleEntityLinker.link
    type_candidates: List[URIRef] = field(default_factory=list)

# class IriMint:
#     """
#     Simple URI minting strategy for new entities.
#     """
#     def __init__(self, base: str):
#         if not base.endswith(("/", "#")):
#             base = base.rstrip("/") + "/"
#         self.base = base

#     def new(self, hint: Optional[str] = None) -> URIRef:
#         # slug from hint, fallback to uuid
#         if hint:
#             slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", hint.strip().lower()).strip("-")
#             if slug:
#                 return URIRef(self.base + slug + "-" + uuid.uuid4().hex[:8])
#         return URIRef(self.base + uuid.uuid4().hex)

def guess_literal(value):
    """
    Heuristic typing: int, float, bool, date-like ISO strings -> XSD types.
    Returns rdflib Literal with datatype or language tag if detected as natural language.
    """
    if isinstance(value, bool):
        return Literal(value)
    if isinstance(value, int):
        return Literal(value)
    if isinstance(value, float):
        return Literal(value)
    if isinstance(value, str):
        # ISO date/datetime quick checks
        iso_date = re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        iso_dt   = re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+\-]\d{2}:\d{2})?", value)
        if iso_dt:
            return Literal(value, datatype=XSD.dateTime)
        if iso_date:
            return Literal(value, datatype=XSD.date)
        # short codes (ids) vs longer labels: keep as plain string
        return Literal(value)
    # fallback to string
    return Literal(str(value))

def get_best_candidate(dict):
    return URIRef(dict["best"]["uri"])

class JsonToRdfLinker:
    """
    Recursively converts a JSON-like structure to RDF using:
      - SimpleEntityLinker for entity nodes
      - SimpleRelationLinker for predicate selection
      - Ontology domains/ranges (soft validation)
    Produces:
      - rdflib.Graph
      - sidecar: list of linking decisions for auditing
    """

    def __init__(
        self,
        kg_path: str,
        ontology_path: str,
        base_ns: str = "http://kg.org/json/",
        soft_validate_domain_range: bool = True,
        mint_new_entities: bool = True,
        add_created_entities_to_index: bool = True,
        dynamic_index: bool = True,
    ):
        self.entity_linker = SimpleEntityLinker(kg_path)
        self.relation_linker = SimpleRelationLinker(ontology_path)
        # self.minter = IriMint(base_ns)
        self.base = Namespace(base_ns)
        self.soft_validate = soft_validate_domain_range
        self.mint = mint_new_entities
        self.add_created = add_created_entities_to_index
        self.dynamic_index = dynamic_index

        # Build a quick lookup: prop_uri -> (domain set, range set)
        self.prop_meta: Dict[str, Dict[str,str]] = {}
        for p in self.relation_linker.properties:
            domain = p.domain.uri if p.domain else "None"
            range = p.range.uri if p.range else "None"
            _type = p.type.name
            self.prop_meta[p.uri] = {"domain": domain, "range": range, "type": _type}

    # ---------- public API ----------

    def link_document(self, obj: dict, parent: URIRef) -> tuple[Graph, List[LinkDecision]]:
        g = Graph()
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("skos", SKOS)
        g.bind("dct", DCTERMS)
        g.bind("ex", self.base)

        decisions: List[LinkDecision] = []
        # root_node, type = 
        self._materialize(g, obj, parent=parent, via_pred=None, decisions=decisions)

        # Mark the root (optional)
        # g.add((root_node, DCTERMS.source, Literal("json-root")))
        return g, decisions

    # ---------- internals ----------

    def _materialize(
        self,
        g: Graph,
        obj,
        parent: URIRef,
        via_pred: Optional[URIRef],
        decisions: List[LinkDecision],
    ) -> URIRef | BNode | None:
        

        if isinstance(obj, dict):

            # TODO if not accept build a new node from tree

            ent_decision = self._link_or_create_entity(obj)
            decisions.append(ent_decision)

            predicate_matches =  self.relation_linker.link(obj)
            label = ent_decision.label


            if label:
                g.add((ent_decision.node, RDFS.label, Literal(label)))
            
            domains = set()
            for match in predicate_matches:
                if match["decision"] == "accept":
                    # instead of range get property type

                    nested_obj = obj[match["query_text"]]

                    best_pred = get_best_candidate(match)

                    if isinstance(nested_obj, dict):
                        self._materialize(g, nested_obj, URIRef(ent_decision.node), best_pred, decisions)
                        return ent_decision.node
                    elif isinstance(nested_obj, list): # process list
                        for item in nested_obj:
                            node = self._materialize(g, item, URIRef(ent_decision.node), best_pred, decisions)
                            if isinstance(node, URIRef):
                                g.add((parent, best_pred, node))
                    else: # pass as literal
                        node = self._materialize(g, nested_obj, URIRef(ent_decision.node), best_pred, decisions)

            return ent_decision.node

        else:
            # Literal
            if parent and via_pred:
                prop_meta = self.prop_meta.get(str(via_pred), {"domain": "None", "range": "None"})
    
                if prop_meta and str(prop_meta["type"]) == "ObjectProperty":
                    print(f"try linking shallow object {obj}")
                    node = self._materialize(g, { "label": obj }, parent, via_pred, decisions)
                    print(node)
                    if isinstance(node, URIRef):
                        g.add((parent, via_pred, node))
                        if prop_meta["range"] and prop_meta["range"] != "None":
                            g.add((node, RDF.type, URIRef(prop_meta["range"])))


                elif prop_meta:
                    if prop_meta["range"] and prop_meta["range"] != "None":
                        literal_value = Literal(obj, None, prop_meta["range"])
                        g.add((parent, via_pred, literal_value))
                    else:
                        literal_value = guess_literal(obj)
                        g.add((parent, via_pred, literal_value))   
                    if prop_meta["domain"] and prop_meta["domain"] != "None":
                        g.add((parent, RDF.type, URIRef(prop_meta["domain"]))) # TODO check if this can be improved
                else:
                    print(f"no prop meta for {str(via_pred)}")
            else:
                raise ValueError(f"Unsupported type: {type(obj)}")




    def _link_or_create_entity(self, obj: dict) -> LinkDecision:

        def hash_obj(obj: dict) -> str:
            return hashlib.md5(json.dumps(obj).encode()).hexdigest()
        
        # Try linking
        link = self.entity_linker.link(obj)
        best = link["best"]
        score = best["score"]

        # Preferred label to keep
        labelish = find_labelish(obj)
        label_val = labelish[1] if labelish else None

        if score >= THRESHOLD:
            node = URIRef(best["uri"])
            # Collect candidate types if the KG already had types (we don’t read KG here; optional extension)
            t_candidates: List[URIRef] = []
            created = False
        else:
            # Mint new entity
            node = URIRef(self.base + hash_obj(obj))
            # if not self.mint:
            #     # If minting disabled, use a blank node
            #     node = BNode()
            # else:
            #     node = self.minter.new(hint=label_val or top_level_kv_string(obj)[:40])

            # Optionally add to index using discovered labels/ids to help future linking
            if self.add_created and self.dynamic_index:
                new_aliases: list[tuple[str, str]] = []
                # Use labelish and any ID-like keys as aliases
                if label_val:
                    new_aliases.append((str(node), label_val))
                for k, v in obj.items():
                    if isinstance(v, str) and (LABEL_KEYS.search(k) or ID_KEYS.search(k)):
                        val = v.strip()
                        if val and len(val) <= 200:
                            new_aliases.append((str(node), val))
                self.entity_linker.update_index(new_aliases)

            created = True
            t_candidates = []

        return LinkDecision(
            node=node,
            created=created,
            label=label_val,
            entity_link=link,
            type_candidates=t_candidates,
        )


# def link_json(json_path: str, kg_path: str, ontology_path: str):
    # linker = SimpleEntityLinker(kg_path)
    # relation_linker = SimpleRelationLinker(ontology_path)
    # obj = json.load(open(json_path))
    # res = linker.link(obj)
    # links = relation_linker.link(obj)
    # return res, links

import os

@Registry.task(
    input_spec={"source": DataFormat.JSON, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Construct RDF graph from JSON using simple entity and relation linking",
    category=["Construction"]
)
def construct_linkedrdf_from_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    json_path = inputs["source"].path
    kg_path = inputs["target"].path

    ontology_path = os.getenv("ONTOLOGY_PATH")
    if not ontology_path:
        raise ValueError("ONTOLOGY_PATH environment variable is not set")

    linker = JsonToRdfLinker(kg_path.as_posix(), ontology_path)
    
    final_g = Graph()

    if os.path.isdir(json_path):
        for file in tqdm(os.listdir(json_path), desc="Processing JSON files"):
            if file.endswith(".json"):
                json_file = os.path.join(json_path, file)
                json_data = json.load(open(json_file))
                g, decisions = linker.link_document(json_data, URIRef("http://kg.org/json/"+ file.split(".")[0])+"/")
                final_g += g
    else:
        json_data = json.load(open(json_path))
        g, decisions = linker.link_document(json_data, URIRef("http://kg.org/json/"+ json_path.name.split(".")[0])+"/")
        final_g += g

    final_g.serialize(format="ntriples", destination=outputs["output"].path)

@Registry.task(
    input_spec={"source": DataFormat.JSON, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Construct RDF graph from JSON using simple entity and relation linking",
    category=["Construction"]
)
def construct_linkedrdf_from_json_v2(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    json_path = inputs["source"].path
    kg_path = inputs["target"].path

    ontology_path = os.getenv("ONTOLOGY_PATH")
    if not ontology_path:
        raise ValueError("ONTOLOGY_PATH environment variable is not set")

    linker = JsonToRdfLinker(kg_path.as_posix(), ontology_path, dynamic_index=False)
    
    final_g = Graph()

    if os.path.isdir(json_path):
        for file in tqdm(os.listdir(json_path), desc="Processing JSON files"):
            if file.endswith(".json"):
                json_file = os.path.join(json_path, file)
                json_data = json.load(open(json_file))
                g, decisions = linker.link_document(json_data, URIRef("http://kg.org/json/"+ file.split(".")[0])+"/")
                final_g += g
    else:
        json_data = json.load(open(json_path))
        g, decisions = linker.link_document(json_data, URIRef("http://kg.org/json/"+ json_path.name.split(".")[0])+"/")
        final_g += g

    final_g.serialize(format="ntriples", destination=outputs["output"].path)

if __name__ == "__main__":    

    # --- Example run ---
    # paths you provide
    KG_TTL = "/home/marvin/project/data/acquisiton/film1k_bundle/split_0/kg/reference/data.nt"
    JSON_PATH = "/home/marvin/project/code/kgflex/src/kgpipe_tasks/test/test_data/json/dbp-movie_depth=1.json"
    OWL_TTL = "/home/marvin/project/data/acquisiton/film1k_bundle/ontology.ttl"

    linker = JsonToRdfLinker(KG_TTL, OWL_TTL)
    json_obj = json.load(open(JSON_PATH))

    g, decisions = linker.link_document(json_obj, URIRef("http://example.org/root/"))

    # Serialize graph
    print(g.serialize(format="turtle"))
    # Inspect decisions (per-object audit trail)
    # for d in decisions:
    #     print(d.node, d.created, d.label, d.entity_link["best"]["score"])
    # linker = SimpleEntityLinker(KG_TTL)
    # obj = json.load(open(JSON_PATH))
    # res = linker.link(obj)
    # print(json.dumps(res, indent=2, ensure_ascii=False))

    # relation_linker = SimpleRelationLinker(OWL_TTL)
    # links = relation_linker.link(obj)
    # print(json.dumps(links, indent=2, ensure_ascii=False))

# import os


# @Registry.task(
#     input_spec={"source": DataFormat.JSON, "target": DataFormat.RDF_NTRIPLES},
#     output_spec={"output": DataFormat.RDF_NTRIPLES},
#     description="Construct RDF graph from JSON",
#     category=["Construction"]
# )
# def construct_linkedrdf_from_json2(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     json_path = inputs["source"].path

#     if os.path.isdir(json_path):
#         for file in os.listdir(json_path):
#             if file.endswith(".json"):
#                 json_path = os.path.join(json_path, file)
#                 json_data = json.load(open(json_path))
#                 linker.update_index(json_data)
#     else:
#         json_data = json.load(open(json_path))
#         linker.update_index(json_data)

    # json = json.load(open("code/kgflex/src/kgpipe_tasks/test/test_data/json/dbp-movie_depth=1.json"))

    # ROLE_SEEDS = {
    #     "label": ["name", "label", "title", "display name", "caption"],
    #     "id":    ["id", "code", "sku", "isbn", "doi", "gtin"],
    #     "desc":  ["description", "summary", "abstract", "comment"],
    # }

    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # # Build role centroids
    # role_vecs = {}
    # for role, seeds in ROLE_SEEDS.items():
    #     vecs = encode(seeds, normalize_embeddings=True)
    #     role_vecs[role] = np.mean(vecs, axis=0)

    # def classify_key(k: str, threshold=0.4):
    #     v = iencode([k], normalize_embeddings=True)[0]
    #     sims = {role: float(v @ rv) for role, rv in role_vecs.items()}
    #     best_role, best_score = max(sims.items(), key=lambda x: x[1])
    #     return best_role if best_score >= threshold else None, sims

    # # Example
    # keys = ["displayLabel", "shortName", "gtin_code", "abstract_text"]
    # for k in keys:
    #     role, sims = classify_key(k)
    #     print(k, "→", role, sims)    # def _materialize_node(
    #     self,
    #     g: Graph,
    #     obj,
    #     parent: Optional[URIRef ],
    #     via_pred: Optional[URIRef],
    #     decisions: List[LinkDecision],
    # ) -> tuple[URIRef | Literal, Optional[URIRef]]:
    #     """
    #     Turns a JSON value into an RDF node and attaches it to parent via via_pred if provided.
    #     Returns (node, rdf_type_if_any)
    #     """
    #     # Literals or shallow entities
    #     if not isinstance(obj, dict) and not isinstance(obj, list):
    #         lit = guess_literal(obj)
    #         if parent is not None and via_pred is not None:
    #             domain, range = self.prop_meta.get(str(via_pred), {"domain": None, "range": None})
    #             if range and not range.startswith(str(XSD)):
    #                 # has to be a node not a literal


    #                 g.add((parent, via_pred, lit))
    #         return lit, None

    #     if isinstance(obj, dict):
    #         relation_links = self.relation_linker.link(obj)
    #         for link in relation_links:
    #             if link["decision"] == "accept":
    #                 via_pred = URIRef(link["best"]["uri"]) # TODO: check if this is correct
                    
    #                 self._materialize_node(g, obj[link["query_text"]], parent, via_pred, decisions)
    #         return , None

    #     # # Arrays – model as repeated predicate edges or rdf:List; we do repeated edges for simplicity
    #     # # if isinstance(obj, list):
    #     # #     # assume parent & via_pred not None; otherwise create a blank node collection
    #     # #     # coll_node = BNode()
    #     # #     # if parent is not None and via_pred is not None:
    #     # #     #     g.add((parent, via_pred, coll_node))
    #     # #     # Link elements with rdf:_n predicates for ordering
    #     # #     for i, item in enumerate(obj, start=1):
    #     # #         # pred = URIRef("http://example.org/resource/" + f"_{i}")
    #     # #         self._materialize_node(g, item, coll_node, pred, decisions)
    #     # #     return coll_node, None
    #     # if isinstance(obj, list):
    #     #     for item in obj:
    #     #         return self._materialize_node(g, item, parent, via_pred, decisions)

    #     # # Objects -> entities
    #     # ent_decision = self._link_or_create_entity(obj)
    #     # node = ent_decision.node
    #     # decisions.append(ent_decision)

    #     # # Label (if present)
    #     # if ent_decision.label:
    #     #     # g.add((node, SKOS.prefLabel, Literal(ent_decision.label)))
    #     #     g.add((node, RDFS.label, Literal(ent_decision.label)))

    #     # # Optional type assertions from ontology/domain hints (if any were collected)
    #     # for t in ent_decision.type_candidates:
    #     #     g.add((node, RDF.type, t))

    #     # # Attach to parent
    #     # if parent is not None and via_pred is not None:
    #     #     g.add((parent, via_pred, node))

    #     # # Now link properties (relations) for this object
    #     # rel_links = self.relation_linker.link(obj.keys())
    #     # rel_by_key = {rl["query_text"]: rl for rl in rel_links}

    #     # for k, v in obj.items():
    #     #     # Choose predicate
    #     #     pred_uri = None
    #     #     if k in rel_by_key:
    #     #         best = rel_by_key[k]["best"]
    #     #         if best["score"] >= THRESHOLD:
    #     #             pred_uri = URIRef(best["uri"])
    #     #         else:
    #     #             continue
    #     #         # else:
    #     #         #     # If low confidence: fallback to data property in example namespace
    #     #         #     pred_uri = self.base[k]
    #     #     else:
    #     #         continue
    #     #         # pred_uri = self.base[k]

    #     #     # Domain/range soft validation
    #     #     pm = self.prop_meta.get(str(pred_uri))
    #     #     if pm and pm["domain"]:
    #     #         # (If you maintain a type index, you could check current types here)
    #     #         for domain in pm["domain"]:
    #     #             g.add((node, RDF.type, domain))

    #     #     # Recurse
    #     #     child_node, child_type = self._materialize_node(g, v, parent=node, via_pred=pred_uri, decisions=decisions)

    #     #     # Range validation — if we know range and child is literal vs resource
    #     #     if pm and pm["range"]:
    #     #         # If range constrained to classes (URIs) but we produced a literal, we may want to lift to a node
    #     #         if isinstance(child_node, Literal):
    #     #             # Optionally: auto-box literal into a value node with ex:value
    #     #             if self.soft_validate:
    #     #                 # lift to value node
    #     #                 # vnode = BNode()
    #     #                 # g.remove((node, pred_uri, child_node))
    #     #                 # g.add((node, pred_uri, vnode))
    #     #                 # g.add((vnode, self.base["value"], child_node))
    #     #                 for range in pm["range"]:
    #     #                     g.add((node, pred_uri, range))
    #     #             else:
    #     #                 # or reject/raise
    #     #                 pass

    #     # return node, None