from __future__ import annotations
import re, uuid
from typing import Any, Dict, Tuple, Optional, Iterable, List
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS, XSD

# --------------------------------------------------------------------------------------
# Helpers: label detection (simple & robust) and the object-vs-literal heuristic
# --------------------------------------------------------------------------------------

LABEL_KEY_RX = re.compile(r"(?:^|[_\-\s])(name|label|title|display\s*name|pref\s*label)$", re.I)

def _norm_key(k: str) -> str:
    k = k.strip()
    k = re.sub(r"([a-z])([A-Z])", r"\1 \2", k)
    return k.replace("-", " ").replace("_", " ").strip()

def find_labelish_value(obj: Dict[str, Any]) -> Optional[str]:
    """
    Return a good label-ish string if present.
    - checks common keys (name/label/title/displayName/prefLabel)
    - supports value being a string or list[str]
    - falls back to None
    """
    # 1) exact-ish keys
    for k, v in obj.items():
        if LABEL_KEY_RX.search(k):
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x.strip():
                        return x.strip()
    # 2) soft fallback: pick the shortest “namey” string field
    candidates: List[str] = []
    for k, v in obj.items():
        if isinstance(v, str) and 1 <= len(v) <= 120:
            candidates.append(v.strip())
        elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            candidates.extend([x.strip() for x in v if 1 <= len(x) <= 120])
    if candidates:
        candidates.sort(key=len)
        return candidates[0]
    return None

RELATION_HINTS = {
    # "director","producer","starring","cast","actor","actress","author","composer",
    # "artist","creator","illustrator","manufacturer","brand","owner","parent",
    # "publisher","distributor","company","organization","org","institution",
    # "country","city","location","place","team","person","people","mentor","advisor"
}
LITERAL_HINTS = {
    # "description","summary","abstract","comment","note","bio","biography",
    # "runtime","duration","length","time","height","weight","price","amount",
    # "count","quantity","rating","score","status","language","lang","category",
    # "genres","genre","keywords","tags","format","type","url","uri","link","email",
    # "phone","isbn","doi","gtin","ean","sku","code","id","date","datetime","year"
}
DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
NUM_RE  = re.compile(r"^-?\d+(\.\d+)?$")
UNIT_RE = re.compile(r"^\s*[-\d\.,]+\s*(cm|mm|km|kg|g|m|s|min|hrs?|ms|gb|mb|kb|°c|°f)\s*$", re.I)
URL_RE  = re.compile(r"^https?://", re.I)
EMAIL_RE= re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
IDISH_RE= re.compile(r"^[A-Z0-9\-_:./]{4,}$")

def looks_like_name(s: str) -> bool:
    tokens = [t for t in re.split(r"\s+", s.strip()) if t]
    cap_tokens = sum(1 for t in tokens if t[:1].isupper())
    return len(tokens) >= 2 and cap_tokens >= 2

def looks_like_enum(s: str) -> bool:
    return 1 <= len(s) <= 20 and re.fullmatch(r"[a-z0-9_\-+/.]+", s) is not None

def key_norm(k: str) -> str:
    k = k.strip().lower().replace("-", "_")
    k = re.sub(r"([a-z])([A-Z])", r"\1_\2", k)
    return k

def heuristic_decide_object_vs_literal(
    key: str,
    value: Any,
    ontology_hint: Optional[str] = None
) -> Tuple[str, float, Dict[str, float]]:
    """
    Returns: (decision, objectness_score, features)
    decision in {"object","literal"}
    """
    k = key_norm(key)

    if ontology_hint in {"object","literal"}:
        return ontology_hint, 1.0, {"ontology_hint": 1.0}

    features: Dict[str, float] = {}

    if isinstance(value, dict):
        features["is_dict"] = 1.0
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            features["list_of_dicts"] = 1.0
        if value and all(isinstance(x, str) for x in value):
            namey = sum(looks_like_name(x) for x in value) / max(len(value), 1)
            features["list_name_ratio"] = namey

    if any(h in k.split("_") for h in RELATION_HINTS):
        features["relation_key_hint"] = 0.7
    if any(h in k.split("_") for h in LITERAL_HINTS):
        features["literal_key_hint"] = 0.7

    if isinstance(value, str):
        if looks_like_name(value):
            features["name_like_value"] = 0.6
        if DATE_RE.match(value):
            features["date_like"] = 0.8
        if NUM_RE.match(value) or UNIT_RE.match(value):
            features["numeric_like"] = 0.8
        if URL_RE.match(value) or EMAIL_RE.match(value) or IDISH_RE.match(value):
            features["id_url_like"] = 0.8
        if len(value) > 80:
            features["long_text"] = 0.6
        if looks_like_enum(value):
            features["enum_like"] = 0.6

    if isinstance(value, (int, float)):
        features["numeric"] = 0.8

    object_pos = (
        features.get("is_dict", 0)
        + features.get("list_of_dicts", 0)
        + features.get("list_name_ratio", 0)
        + features.get("relation_key_hint", 0)
        + features.get("name_like_value", 0)
    )
    literal_pos = (
        features.get("literal_key_hint", 0)
        + features.get("date_like", 0)
        + features.get("numeric_like", 0)
        + features.get("id_url_like", 0)
        + features.get("long_text", 0)
        + features.get("enum_like", 0)
        + features.get("numeric", 0)
    )
    score = object_pos / (object_pos + literal_pos + 1e-9)
    decision = "object" if score >= 0.5 else "literal"
    return decision, float(score), features

# --------------------------------------------------------------------------------------
# JSON → RDF with heuristics
# --------------------------------------------------------------------------------------

EXS = Namespace("http://ex.com/source/")     # base
EXP = Namespace("http://ex.com/source/p/")   # properties (semantic / canonical)
EXPL= Namespace("http://ex.com/source/pl/")  # properties for literal verbatim (parallel)
EXC = Namespace("http://ex.com/source/class/")

def _mint_entity_uri(kind: str) -> URIRef:
    return URIRef(f"{EXS}ent/{kind}/{uuid.uuid4()}")

def _prop_uri(k: str) -> URIRef:
    return URIRef(f"{EXP}{k}")

def _prop_lit_uri(k: str) -> URIRef:
    return URIRef(f"{EXPL}{k}")

def _literal(v: Any) -> Literal:
    if isinstance(v, bool):
        return Literal(v)
    if isinstance(v, int):
        return Literal(v, datatype=XSD.integer)
    if isinstance(v, float):
        return Literal(v, datatype=XSD.decimal)
    return Literal(str(v))

def json_to_rdf(obj: Dict[str, Any], root_kind: str = "Root") -> Tuple[Graph, URIRef]:
    """
    Builds a richer RDF:
      - root gets rdfs:label/skos:prefLabel from find_labelish_value
      - for each key:
          * if heuristic says LITERAL → :p/<key> literal
          * if OBJECT:
              - emit :pl/<key> with the verbatim literal(s), for provenance/compat
              - mint node(s) with rdfs:label and link via :p/<key>
    """
    g = Graph()
    g.bind("exs", EXS); g.bind("exp", EXP); g.bind("expl", EXPL)
    g.bind("rdfs", RDFS); g.bind("skos", SKOS)

    def add_entity(o: Dict[str, Any], kind: str, generate_type: bool = False) -> URIRef:
        base = "http://example.com/test/"
        s = URIRef(base + root_kind) #_mint_entity_uri(kind)
        if generate_type:
            g.add((s, RDF.type, URIRef(f"{EXC}{kind}")))

        # Root/entity labels
        lab = find_labelish_value(o)
        if lab:
            g.add((s, RDFS.label, Literal(lab)))
            # g.add((s, SKOS.prefLabel, Literal(lab)))

        for k, v in o.items():
            k_norm = re.sub(r"\s+", "_", _norm_key(k).lower())
            p_obj  = _prop_uri(k_norm)
            p_lit  = _prop_lit_uri(k_norm)

            # Decide based on heuristic
            decision, score, _ = heuristic_decide_object_vs_literal(k, v)

            # Case 1: literal-ish
            if decision == "literal":
                if isinstance(v, list):
                    for item in v:
                        if item is not None and not isinstance(item, dict):
                            g.add((s, p_obj, _literal(item)))
                elif not isinstance(v, dict):
                    g.add((s, p_obj, _literal(v)))
            
            else:
                # Case 2: object-ish
                if isinstance(v, dict):
                    o2 = add_entity(v, kind=root_kind + "_" + k_norm.capitalize())
                    g.add((s, p_obj, o2))
                elif isinstance(v, list):
                    # emit verbatim literals for provenance/compat, and mint nodes where possible
                    for item in v:
                        if isinstance(item, dict):
                            o2 = add_entity(item, kind=root_kind + "_" + k_norm.capitalize())
                            g.add((s, p_obj, o2))
                        elif isinstance(item, (str, int, float)) and str(item).strip():
                            # provenance literal
                            # g.add((s, p_lit, _literal(item)))
                            # mint a node with label if it looks like an entity-ish string
                            lab_str = str(item).strip()
                            o2 = URIRef(base + root_kind + "_" + k_norm.capitalize()) #_mint_entity_uri(root_kind + "_" + k_norm.capitalize())
                            g.add((o2, RDFS.label, Literal(lab_str)))
                            
                            if generate_type:
                                g.add((o2, RDF.type, URIRef(f"{EXC}{k_norm.capitalize()}")))    
                            # g.add((o2, SKOS.prefLabel, Literal(lab_str)))
                            g.add((s, p_obj, o2))
                else:
                    # single primitive that looks like an entity name → dual representation
                    if v is not None and str(v).strip():
                        # g.add((s, p_lit, _literal(v)))  # keep raw literal
                        lab_str = str(v).strip()
                        o2 = URIRef(base + root_kind + "_" + k_norm.capitalize()) #_mint_entity_uri(root_kind + "_" + k_norm.capitalize())
                        g.add((o2, RDFS.label, Literal(lab_str)))
                        if generate_type:
                            g.add((o2, RDF.type, URIRef(f"{EXC}{k_norm.capitalize()}")))    
                        # g.add((o2, SKOS.prefLabel, Literal(lab_str)))
                        g.add((s, p_obj, o2))

        return s

    root = add_entity(obj, root_kind)
    return g, root

import json, os
from kgpipe.common import Registry, DataFormat, Data

@Registry.task(
    input_spec={"source": DataFormat.JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Construct RDF graph from JSON",
    category=["Construction"]
)
def construct_rdf_from_json2(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    file_or_dir_path = inputs["source"].path
    if os.path.isdir(file_or_dir_path):
        final_g = Graph()
        # os.makedirs(outputs["output"].path, exist_ok=True)
        for file in os.listdir(file_or_dir_path):
            if file.endswith(".json"):
                json_path = os.path.join(file_or_dir_path, file)
                json_data = json.load(open(json_path))
                g, root = json_to_rdf(json_data, file.split(".")[0])
                final_g += g
        final_g.serialize(format="nt", destination=outputs["output"].path)
    else:
        json_data = json.load(open(file_or_dir_path))
        g, root = json_to_rdf(json_data)
        g.serialize(format="nt", destination=outputs["output"].path)

if __name__ == "__main__":
    json = json.load(open("code/kgflex/src/kgpipe_tasks/test/test_data/json/dbp-movie_depth=1.json"))
    g, root = json_to_rdf(json)
    print(g.serialize(format="ttl"))