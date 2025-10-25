from typing import Dict, List, Tuple
import json
from kgpipe.common import Data, DataFormat, Registry

from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document, TE_Triple, TE_Span, TE_Pair
# from kgpipe_tasks.text_processing import AliasAndTransformerBasedRelationLinker, RelationMatch

from typing import Dict, List
import json

import hashlib
from tqdm import tqdm
import os




def __getSubject(data):
    # Ensure stable serialization for consistent hashing
    hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return f"http://example.org/{hash}"

def __extract_data(data, parent_key=None, parent_subject=None):
    """
    { name: foo, genre:[], involved: [{}]}
    """
    triplets = []

    # value can be dict, list, list_of_dicts, value
    if isinstance(data, dict):
        subject = __getSubject(data)
        if parent_subject:
            triplets.append((parent_subject, parent_key, subject))
        for key, value in data.items():
            triplets += __extract_data(value, key, subject)
    elif isinstance(data, list):
        for item in data:
            # triplets.append((parent_subject,parent_key,subject))
            triplets += __extract_data(item, parent_key, parent_subject)

    else:
        # Primitive value: attach to parent
        triplets.append((parent_subject, parent_key, data))
                            
    return triplets


# def __extract_json(json_data):

#     triplets =  __extract_data(json_data)
#     triplets_spans = []

#     for t in triplets:
#         if str(t[0]).startswith("http"):
#             head = TE_Span(start=0, end=len(t[0]), surface_form=t[0], mapping=t[0])
#         else:
#             head = TE_Span(start=0, end=len(t[0]), surface_form=t[0], text=t[0])

#         if str(t[1]).startswith("http"):
#             rel = TE_Span(start=0, end=len(t[1]), surface_form=t[1], mapping=t[1])
#         else:
#             rel = TE_Span(start=0, end=len(t[1]), surface_form=t[1], text=t[1])

#         if str(t[2]).startswith("http"):
#             tail = TE_Span(start=0, end=len(t[2]), surface_form=t[2], mapping=t[2])
#         else:
#             tail = TE_Span(start=0, end=len(t[2]), surface_form=t[2], text=t[2])

#         triplets_spans.append(TE_Triple(subject=head, predicate=rel, object=tail))

#     te_doc = TE_Document(text="",triples=triplets_spans)
#     return te_doc

def __extract_data_filenameUri(json_data, filename):

    triplets = __extract_json_data_filenameUri(json_data, parent_subject=f"http://kg.org/json/{filename.split('.')[0]}")

    # for t in triplets:
    #     print(t)


    triplets_spans = []

    for t in triplets:
        if str(t[0]).startswith("http"):
            head = TE_Span(start=0, end=len(t[0]), surface_form=t[0], mapping=t[0])
        else:
            head = TE_Span(start=0, end=len(t[0]), surface_form=t[0], text=t[0])

        if str(t[1]).startswith("http"):
            rel = TE_Span(start=0, end=len(t[1]), surface_form=t[1], mapping=t[1])
        else:
            rel = TE_Span(start=0, end=len(t[1]), surface_form=t[1], text=t[1])

        if str(t[2]).startswith("http"):
            tail = TE_Span(start=0, end=len(t[2]), surface_form=t[2], mapping=t[2])
        else:
            tail = TE_Span(start=0, end=len(t[2]), surface_form=t[2], text=t[2])

        triplets_spans.append(TE_Triple(subject=head, predicate=rel, object=tail))

    te_doc = TE_Document(text="",triples=triplets_spans)
    return te_doc

def __extract_json_data_filenameUri(json_data, parent_key=None, parent_subject=None):
    """
    { name: foo, genre:[], involved: [{}]}
    """
    triplets = []

    # value can be dict, list, list_of_dicts, value
    if isinstance(json_data, dict):
        # subject = __getSubject(json_data)
        # if parent_subject:
        #     triplets.append((parent_subject, parent_key, subject))
        for key, value in json_data.items():
            triplets += __extract_data(value, key, parent_subject)
    elif isinstance(json_data, list):
        for item in json_data:
            # triplets.append((parent_subject,parent_key,subject))
            triplets += __extract_data(item, parent_key, parent_subject)

    else:
        # Primitive value: attach to parent
        triplets.append((parent_subject, parent_key, json_data))
                            
    return triplets



# def triplify_json_dm(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     json_data = json.load(open(inputs["source"].path))
#     te_doc = __extract_json(json_data)
#     graph = __generateRDF(te_doc)
#     graph.serialize(outputs["output"].path, format="nt")
#     print(f"RDF written to {outputs['output'].path}")


# triplify_json_dm_task = KgTask(
#     name="triplify_json_dm",
#     input_spec={"source": DataFormat.JSON},
#     output_spec={"output": DataFormat.RDF_NTRIPLES},
#     function=triplify_json_dm,
#     category="DataMapping"
#     # name="triplify_json_dm",
#     # inputDict={"source": JSON},
#     # outputDict={"output": IE_JSON},
#     # function=triplify_json_dm,
#     # task_category=[DataMapping]
# )

# def triplify_json_dir_dm(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     dir = inputs["source"].path

#     os.makedirs(outputs["output"].path, exist_ok=True)

#     for file in tqdm.tqdm(os.listdir(dir)):
#         if file.endswith(".json"):
#             json_data = json.load(open(os.path.join(dir, file)))
#             te_doc = __extract_json(json_data)
#             with open(os.path.join(outputs["output"].path, file), 'w') as f:
#                 f.write(te_doc.model_dump_json())
#     print(f"TE_DOCs written to {outputs['output'].path}")

# triplify_json_dir_dm_task = KgTask(
#     name="triplify_json_dir_dm",
#     input_spec={"source": DataFormat.JSON},
#     output_spec={"output": DataFormat.IE_JSON},
#     function=triplify_json_dir_dm,
#     category="DataMapping"
# )

def triplify_json_dir__with_filename_uri_dm(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    dir = inputs["source"].path
    os.makedirs(outputs["output"].path, exist_ok=True)
    for file in tqdm(os.listdir(dir), desc="Triplifying JSON files"):
        if file.endswith(".json"):
            json_data = json.load(open(os.path.join(dir, file)))
            te_doc = __extract_data_filenameUri(json_data, file)
            with open(os.path.join(outputs["output"].path, file), 'w') as f:
                f.write(te_doc.model_dump_json())
    print(f"TE_DOCs written to {outputs['output'].path}")

# triplify_json_dir__with_filename_uri_dm_task = KgTask(
#     name="triplify_json_dir__with_filename_uri_dm",
#     input_spec={"source": DataFormat.JSON},
#     output_spec={"output": DataFormat.IE_JSON},
#     function=triplify_json_dir__with_filename_uri_dm,
#     category="DataMapping"
# )   


# def test_json_extract():
    
#     json_data = json.loads(open("/home/marvin/project/data/current/sources/source.json/0a0d9846a69456249a3261102d4fd534.json").read())

#     # print(json_data)

#     rl = AliasAndTransformerBasedRelationLinker("/home/marvin/project/data/current/ontology.ttl")

#     triplets = __extract_triplets_from_json(json_data, "0a0d9846a69456249a3261102d4fd534", "root")

#     # print(doc.model_dump_json(indent=2))

#     doc = TE_Document(text="", triples=triplets)

#     matches = __link_relations(doc, rl)
#     match_map = {match.relation: match for match in matches}

#     new_triples = []

#     for triplets in doc.triples:
#         if triplets.predicate.text is not None:
#             best_match = match_map[triplets.predicate.text]
#             if best_match.score > 0.5:
#                 p = TE_Span(text=triplets.predicate.text, start=0, end=len(triplets.predicate.text), mapping=best_match.predicate.uri)
#                 new_triples.append(TE_Triple(subject=triplets.subject, predicate=p, object=triplets.object))
#             else:
#                 new_triples.append(triplets)
#         else:
#             new_triples.append(triplets)

#     doc.triples = new_triples

#     print(doc.model_dump_json(indent=2))


# if __name__ == "__main__":
#     test_json_extract()

# #######################################


# #######################################

# def generate_json_schema():
#     """
#     TODO
#     """
#     json_data = {"age": 22, "available": True}

#     DynamicModel = create_model(
#         'DynamicModel',
#         **{key: (type(value), ...) for key, value in json_data.items()}
#     )

#     # Validate or parse data
#     instance = DynamicModel(**json_data)
#     print(instance)

# # TODO used to convert
# def __generic_csv_to_rdf(file_path, namespace):
#     csv = pd.read_csv(file_path)

#     # TODO find id fields

#     g = Graph()
#     for index, row in csv.iterrows():
#         g.add((URIRef(f"{namespace}{row[0]}"), URIRef(f"{namespace}{row[1]}"), URIRef(f"{namespace}{row[2]}")))

#     return g

# if __name__ == "__main__":
#     JSON_STR="""
#     {
#         "budget": "650000.0",
#         "cinematography": "Richard Careaga",
#         "director": [
#             {
#                 "birthDate": "1966-06-09",
#                 "birthPlace": "Asunci\u00f3n, Paraguay",
#                 "name": "Juan Carlos Maneglia",
#                 "nationality": "Paraguayan",
#                 "occupation": "Film director, screenwriter"
#             },
#             "Tana Sch\u00e9mbori"
#         ],
#         "gross": "1000000.0",
#         "name": "7",
#         "producer": [
#             "Camilo Ramirez Guanes",
#             "Vicky Jou"
#         ],
#         "runtime": "6600.0"
#     }
#     """

#     # json_data = json.loads(JSON_STR)
#     # te_doc = __extract_json(json_data)
#     # graph = generateRDF(te_doc, generic=True)
#     # print(graph.serialize(format="ttl"))\

# TODO
@Registry.task(
    input_spec={"source": DataFormat.JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Construct RDF graph from JSON",
    category=["Construction"]
)
def construct_rdf_from_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    pass

@Registry.task(
    input_spec={"source": DataFormat.JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Construct TE_Document from JSON",
    category=["Construction"]
)
def construct_te_document_from_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    triplify_json_dir__with_filename_uri_dm(inputs, outputs)


def __construct_te_triples_from_json_file(json_file: str):
    json_data = json.load(open(json_file))

    def best_label_key(keys):
        """
        get the dict key best identifying the name or label of the json object
        """
        best_key = None
        best_score = 0
        for key in keys:
            if key in json_data:
                score = 0
                if key == "name":
                    score = 1
                elif key == "title":
                    score = 0.9
        return best_key
    
    
    return te_doc.triples

@Registry.task(
    input_spec={"source": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Construct RDF graph from TE_Document",
    category=["Construction"]
)
def construct_te_triples_from_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    pass

# --- temporary section ---

# simple_baseline.py
from typing import Iterator
import re
import json
import numpy as np
from rdflib import Graph
from rdflib.namespace import SKOS, RDFS
from sentence_transformers import SentenceTransformer

# --- Config ---
LABEL_KEYS = re.compile(r"(?:^|_)(name|label|title|display.?name|prefLabel)$", re.I)
ID_KEYS    = re.compile(r"(?:^|_)(id|code|sku|gtin|ean|isbn|doi|cas)$", re.I)
EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD  = 0.70  # tune later

# --- Helpers ---
def find_labelish(obj: dict) -> Iterator[tuple[str, str]]:
    """
    return (key, value) if key and value else None
    """
    for k, v in obj.items():
        if isinstance(v, str) and LABEL_KEYS.search(k) and 1 <= len(v) <= 200:
            yield k, v.strip()

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
    def __init__(self, items, model):
        self.uris  = [u for u,_ in items]
        self.texts = [t for _,t in items]
        self.model = model
        self.vecs  = normalize(self.model.encode(self.texts, convert_to_numpy=True))
    def search(self, query_text: str, k: int = 10):
        q = normalize(self.model.encode([query_text], convert_to_numpy=True))[0]
        sims = self.vecs @ q  # cosine (since normalized)
        idx = np.argsort(-sims)[:k]
        return [(self.uris[i], self.texts[i], float(sims[i])) for i in idx]

# --- Baseline linker ---
class SimpleLinker:
    def __init__(self, ttl_path: str):
        self.model = SentenceTransformer(EMB_MODEL)
        items = load_kg_labels(ttl_path)
        self.index = LabelIndex(items, self.model)

    def link(self, obj: dict):
        label = find_labelish(obj)
        if label:
            query = label
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

# --- Example run ---
# if __name__ == "__main__":
#     # paths you provide
#     KG_TTL = "examples/demo_kg.ttl"
#     JSON_PATH = "examples/demo.json"

#     linker = SimpleLinker(KG_TTL)
#     obj = json.load(open(JSON_PATH))
#     res = linker.link(obj)
#     print(json.dumps(res, indent=2, ensure_ascii=False))


# from sentence_transformers import SentenceTransformer
# import numpy as np

# if __name__ == "__main__":

#     ROLE_SEEDS = {
#         "label": ["name", "label", "title", "display name", "caption"],
#         "id":    ["id", "code", "sku", "isbn", "doi", "gtin"],
#         "desc":  ["description", "summary", "abstract", "comment"],
#     }

#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     # Build role centroids
#     role_vecs = {}
#     for role, seeds in ROLE_SEEDS.items():
#         vecs = model.encode(seeds, normalize_embeddings=True)
#         role_vecs[role] = np.mean(vecs, axis=0)

#     def classify_key(k: str, threshold=0.4):
#         v = model.encode([k], normalize_embeddings=True)[0]
#         sims = {role: float(v @ rv) for role, rv in role_vecs.items()}
#         best_role, best_score = max(sims.items(), key=lambda x: x[1])
#         return best_role if best_score >= threshold else None, sims

#     # Example
#     keys = ["displayLabel", "shortName", "gtin_code", "abstract_text"]
#     for k in keys:
#         role, sims = classify_key(k)
#         print(k, "→", role, sims)

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS, XSD
import uuid, json

EXS = Namespace("http://ex.com/source/")
EXT = Namespace("http://ex.com/target/")

def json_to_rdf(obj, base="http://ex.com/source/ent/"):
    g = Graph()
    def mk_uri(kind): return URIRef(f"{base}{kind}/{uuid.uuid4()}")
    def add_entity(o, kind="Entity"):
        s = mk_uri(kind)
        g.add((s, RDF.type, EXS[kind]))
        for k, v in o.items():
            if isinstance(v, (str, int, float)):
                p = URIRef(f"http://ex.com/source/p/{k}")
                lit = Literal(v)
                g.add((s, p, lit))
                # heuristics for labels/ids
                if k.lower() in {"name","label","title"} and isinstance(v, str):
                    g.add((s, RDFS.label, Literal(v)))
                    g.add((s, SKOS.prefLabel, Literal(v)))
            elif isinstance(v, dict):
                o2 = add_entity(v, kind=k.capitalize())
                p = URIRef(f"http://ex.com/source/p/{k}")
                g.add((s, p, o2))
            elif isinstance(v, list):
                p = URIRef(f"http://ex.com/source/p/{k}")
                for item in v:
                    if isinstance(item, dict):
                        o2 = add_entity(item, kind=k.capitalize())
                        g.add((s, p, o2))
                    else:
                        g.add((s, p, Literal(item)))
        return s
    root = add_entity(obj, kind="Root")
    return g, root

import re
from typing import Any, Dict, Tuple

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

DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")            # 2024 / 2024-05 / 2024-05-12
NUM_RE  = re.compile(r"^-?\d+(\.\d+)?$")                      # number
UNIT_RE = re.compile(r"^\s*[-\d\.,]+\s*(cm|mm|km|kg|g|m|s|min|hrs?|ms|gb|mb|kb|°c|°f)\s*$", re.I)
URL_RE  = re.compile(r"^https?://", re.I)
EMAIL_RE= re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
IDISH_RE= re.compile(r"^[A-Z0-9\-_:./]{4,}$")                 # codes/ids-ish

def looks_like_name(s: str) -> bool:
    # Multi-token with significant capitalization: "Iris Lancaster", "Oliver Drake (filmmaker)"
    tokens = [t for t in re.split(r"\s+", s.strip()) if t]
    cap_tokens = sum(1 for t in tokens if t[:1].isupper())
    return len(tokens) >= 2 and cap_tokens >= 2

def looks_like_enum(s: str) -> bool:
    # short, mostly lowercase/alnum/hyphen/underscore
    return 1 <= len(s) <= 20 and re.fullmatch(r"[a-z0-9_\-+/.]+", s) is not None

def key_norm(k: str) -> str:
    k = k.strip().lower().replace("-", "_")
    k = re.sub(r"([a-z])([A-Z])", r"\1_\2", k)
    return k

def heuristic_decide_object_vs_literal(key: str, value: Any, ontology_hint: str | None = None
                                      ) -> Tuple[str, float, Dict[str, float]]:
    """
    Returns: (decision, objectness_score, features)
    decision in {"object","literal"}
    objectness_score in [0,1]
    """
    k = key_norm(key)

    # Ontology hint: "object" or "literal" if you already know range
    if ontology_hint in {"object","literal"}:
        return ontology_hint, 1.0, {"ontology_hint": 1.0}

    features: Dict[str, float] = {}

    # 1) Type-based priors
    if isinstance(value, dict):
        features["is_dict"] = 1.0
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            features["list_of_dicts"] = 1.0
        # list of strings that look like names?
        if value and all(isinstance(x, str) for x in value):
            namey = sum(looks_like_name(x) for x in value) / max(len(value), 1)
            features["list_name_ratio"] = namey  # 0..1

    # 2) Key-name hints
    if any(h in k.split("_") for h in RELATION_HINTS):
        features["relation_key_hint"] = 0.7
    if any(h in k.split("_") for h in LITERAL_HINTS):
        features["literal_key_hint"] = 0.7

    # 3) Value-shape hints (for primitive or single string)
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

    # 4) Combine into an "objectness" score
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

    # Soft normalization
    score = object_pos / (object_pos + literal_pos + 1e-9)

    decision = "object" if score >= 0.5 else "literal"
    return decision, float(score), features


if __name__ == "__main__":
    json = json.load(open("code/kgflex/src/kgpipe_tasks/test/test_data/json/dbp-movie_depth=1.json"))
    # g, root = json_to_rdf()

    # print(root)
    # print(g.serialize(format="ttl"))

    labels = find_labelish(json)
    for key, value in labels:
        print("LABEL",key, value)


    for key, value in json.items():


        # if not isinstance(value, list) and not isinstance(value, dict):

        decision, score, features = heuristic_decide_object_vs_literal(key, value)
        print(key, value, decision, score, features)
        # elif not isinstance(value, dict):
        #     decision, score, features = heuristic_decide_object_vs_literal(key, value)
        #     print(key, value, decision, score, features)





# def generate_triple_from_json(json_data, parent_s: str, parent_p: str) -> List[Tuple[str, str, str]]:
#     triplets = []

#     if isinstance(json_data, Dict):
#         for key, value in json_data.items():
#             triplets.extend(generate_triple_from_json(value, parent_s, key))
#     elif isinstance(json_data, List):
#         for item in json_data:
#             triplets.extend(generate_triple_from_json(item, parent_s, parent_p))
#     else:
#         triplets.append((parent_s, parent_p, json_data))

#     return triplets

# # def test_json_map():
# #     json_data = json.loads(open("/home/marvin/project/data/current/sources/source.json/0a0d9846a69456249a3261102d4fd534.json").read())

# #     triplets = generate_triple_from_json(json_data, "0a0d9846a69456249a3261102d4fd534", "root")

# #     print(triplets)


# # JSONb
# # TODO

# # JSONb
# # 1. JSON to TE_Document
# # 2. ReL on TE_Document
# # 3. TE_Document to KG

# # Idea:
# # For each key in json find a mapping to the ontology

# def create_triple(s: str, p: str, o: str) -> TE_Triple:
#     print(s, p, o)
#     return TE_Triple(subject=TE_Span(text=s, start=0, end=len(s)), predicate=TE_Span(text=p, start=0, end=len(p)), object=TE_Span(text=o, start=0, end=len(o)))

# # TODO see mapping_tasks.py triplify_json_dm
# def __extract_triplets_from_json(json_data, parent_s: str, parent_p: str) -> List[TE_Triple]:
#     triplets = []

#     if isinstance(json_data, Dict):
#         for key, value in json_data.items():
#             triplets.extend(__extract_triplets_from_json(value, parent_s, key))
#     elif isinstance(json_data, List):
#         for item in json_data:
#             triplets.extend(__extract_triplets_from_json(item, parent_s, parent_p))
#     else:
#         triplets.append(create_triple(parent_s, parent_p, json_data))

#     # print(triplets)
    
#     return triplets

# def __link_relations(doc: TE_Document, rel_matcher: AliasAndTransformerBasedRelationLinker) -> List[RelationMatch]:
#     relations : List[str] = list({ triple.predicate.text for triple in doc.triples if triple.predicate.text is not None})

#     # print(relations)

#     matches = rel_matcher.link_relations(relations)

#     return matches
#     # for match in matches:
#     #     print(match.relation, match.predicate.equivalent, match.score)


# def derive_triple_from_document(doc: TE_Document) -> List[TE_Triple]:
#     return [triple for triple in doc.triples if triple.predicate.text is not None]
