from kgpipe.common.models import KG
import json
from pathlib import Path
import pandas as pd
import csv  
import pyodibel
from rdflib import RDFS, URIRef, Graph
from dataclasses import dataclass
from kgpipe.util.embeddings.st_emb import get_model
import numpy as np
from pyodibel.datasets.mp_mf.multipart_multisource import read_entities_csv

# model

# entity dict
# {
#     "entity_id_1": { "label": "entity_label_1", "type": "entity_type_1" },
#     "entity_id_2": { "label": "entity_label_2", "type": "entity_type_2" },
#     "entity_id_3": { "label": "entity_label_3", "type": "entity_type_3" },
# }\

DEBUG = False

SOFT_THRESHOLD = 0.95

@dataclass
class EntityCoverageResult:
    expected_entities_count: int
    found_entities_count: int
    overlapping_entities_count: int

@dataclass
class TripleAlignmentResult:
    actual_triples_count: int
    reference_triples_count: int
    aligned_triples_count: int

@dataclass
class BinaryClassificationResult:
    tp: int
    fp: int
    tn: int
    fn: int
    
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)
    
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)
    
    def f1_score(self) -> float:
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def __str__(self):
        return f"tp: {self.tp}, fp: {self.fp}, tn: {self.tn}, fn: {self.fn}, accuracy: {self.accuracy()}, precision: {self.precision()}, recall: {self.recall()}, f1_score: {self.f1_score()}"

    def __dict__(self):
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score()
        }

# utility functions

def load_entity_dict_from_json(path: Path) -> dict:
    """
    """
    return json.load(open(path, "r"))

def load_entity_dict_from_csv(path: Path, delimiter: str = ",") -> dict:
    """
    id, type, label
    """
    entity_dict = {}
    with open(path, "r") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if line:
                splits = line.split(delimiter)
                # the csv can only have one, two or three columns
                if len(splits) == 1:
                    id = splits[0]
                    entity_dict[id] = { "label": id, "type": None }
                elif len(splits) == 2:
                    id, label = splits
                    entity_dict[id] = { "label": label, "type": None }
                elif len(splits) == 3:
                    id, label, type = splits
                    entity_dict[id] = { "label": label, "type": type }
                else:
                    raise ValueError(f"Invalid line: {line}")

    return entity_dict

def load_entity_dict(path: Path) -> dict:
    """
    """
    if path.name.endswith(".json"):
        return load_entity_dict_from_json(path)
    elif path.name.endswith(".csv"):
        return {entity.entity_id: entity for entity in read_entities_csv(path)}
    else:
        raise ValueError(f"Unsupported file type: {path}")

# public functions for source entity coverage

def evaluate_source_entity_coverage(kg: KG, entity_dict_path: Path) -> EntityCoverageResult:
    """
    return 0.0
    """
    entity_dict = load_entity_dict(entity_dict_path)
    entity_uris = set(entity_dict.keys())
    entity_labels = set([entity_dict[uri].entity_label for uri in entity_dict if entity_dict[uri].entity_label is not None])

    found_entities = set()
    overlapping_entities = set()

    graph = kg.get_graph()
    for s, p, label in graph.triples((None, RDFS.label, None)):
        found_entities.add(str(s))
        if str(s) in entity_uris or str(label) in entity_labels:
            overlapping_entities.add(str(s))

    return EntityCoverageResult(
        expected_entities_count=len(entity_uris),
        found_entities_count=len(found_entities),
        overlapping_entities_count=len(overlapping_entities)
    )


def evaluate_source_entity_coverage_fuzzy(kg: KG, entity_dict_path: Path) -> EntityCoverageResult:
    """
    checks expected & integrated source entity overlap using label embeddings
    """
    model = get_model()
    entity_dict = load_entity_dict(entity_dict_path)
    entity_labels = list(set([entity_dict[uri].entity_label for uri in entity_dict if entity_dict[uri].entity_label is not None]))
    entity_labels_embeddings = model.encode(entity_labels, convert_to_numpy=True, show_progress_bar=False)

    found_labeles = []
    overlapping_entities = set()

    graph = kg.get_graph()
    for s, p, label in graph.triples((None, RDFS.label, None)):
        found_labeles.append(str(label))

    found_labels_embeddings = model.encode(found_labeles, convert_to_numpy=True, show_progress_bar=False)

    matches = np.dot(found_labels_embeddings, entity_labels_embeddings.T)
    matches = [any(score >= SOFT_THRESHOLD for score in match) for match in matches]

    for idx, is_match in enumerate(matches):
        if is_match:
            overlapping_entities.add(found_labeles[idx])

    if DEBUG:
        missing_entities = set(entity_labels) - overlapping_entities
        print(f"Missing entities: {missing_entities}")

    return EntityCoverageResult(
        expected_entities_count=len(entity_dict),
        found_entities_count=len(found_labeles),
        overlapping_entities_count=len(overlapping_entities)
    )


def evaluate_source_entity_coverage_with_paris(kg: KG, entity_dict_path: Path) -> float:
    return 0.0

# public functions for reference kg aligment

def evaluate_reference_triple_alignment(kg: KG, reference_kg: KG) -> TripleAlignmentResult:
    """
    """
    actual_triples = set()
    reference_triples = set()
    graph = kg.get_graph()
    for s, p, o in graph.triples((None, None, None)):
        actual_triples.add(f"{s} {p} {o}")
    graph = reference_kg.get_graph()
    for s, p, o in graph.triples((None, None, None)):
        reference_triples.add(f"{s} {p} {o}")

    actual_triples = set(actual_triples)
    reference_triples = set(reference_triples)
    aligned_triples = actual_triples & reference_triples

    return TripleAlignmentResult(
        actual_triples_count=len(actual_triples),
        reference_triples_count=len(reference_triples),
        aligned_triples_count=len(aligned_triples) # TODO: check if this is correct
    )

def fuzzy_match(test_values, reference_values, model, threshold=0.9):
    test_values_embeddings = model.encode(test_values, convert_to_numpy=True, show_progress_bar=False)
    reference_values_embeddings = model.encode(reference_values, convert_to_numpy=True, show_progress_bar=False)
    
    matches =np.dot(test_values_embeddings, reference_values_embeddings.T)
    # [[1.         0.91628873 0.92775965 0.7897815 ], first test_value
    # [0.6234202  0.5606477  0.60967    0.5370069 ]], second test_value
    return [any(score >= threshold for score in match) for match in matches]

def evaluate_reference_triple_alignment_fuzzy(test_graph: Graph, ref_graph: Graph) -> BinaryClassificationResult:

    test_sp = test_graph.subject_predicates()

    model = get_model()

    def resolve_uris_as_labels(objs):
        for obj in objs:
            if isinstance(obj, URIRef):
                values = list(ref_graph.objects(obj, RDFS.label))
                for value in values:
                    yield str(value)
            else:
                yield str(obj)

    def check_obj_alignment(test_objs, reference_objs):
        test_values = list(set(resolve_uris_as_labels(test_objs)))
        reference_values = list(set(resolve_uris_as_labels(reference_objs)))
        return fuzzy_match(test_values, reference_values, model)

    # tp aligned triples (covered by reference  )
    # fp unknown triples (not covered by reference)
    # tn 
    # fn missing triples (covered by reference)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    checked_sp = set()

    for s, p in test_sp:
        ref_objs = list(ref_graph.objects(s, p))

        if len(ref_objs) > 0 and not (s, p) in checked_sp:
            checked_sp.add((s, p))
            test_objs = list(test_graph.objects(s, p))

            for idx, is_match in enumerate(check_obj_alignment(test_objs, ref_objs)):
                if is_match:
                    # print(f"tp: {s} {p} {test_objs[idx]}")
                    tp += 1
                else:
                    # print(f"fp: {s} {p} {test_objs[idx]}")
                    fp += 1

        else:
            #print(f"fp: {s} {p}")
            fp += 1

    # calculate fn
    reference_sp = ref_graph.subject_predicates()
    for s, p in reference_sp:
        test_objs = list(test_graph.objects(s, p))
        if len(test_objs) == 0:
            #print(f"fn: {s} {p}")
            fn += 1

    return BinaryClassificationResult(tp=tp, fp=fp, tn=tn, fn=fn)


def evaluate_reference_triple_alignment_with_paris(kg: KG, reference_kg: KG) -> float:
    return 0.0


# def _extract_triples(self, kg: KG) -> Set[str]:
#     """Extract triples from KG file."""
#     triples = set()
    
#     if kg.format.value in ['ttl', 'rdf', 'jsonld']:
#         with open(kg.path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if line and not line.startswith('#'):
#                     parts = line.split()
#                     if len(parts) >= 3:
#                         # Create triple representation
#                         triple = f"{parts[0]} {parts[1]} {parts[2]}"
#                         triples.add(triple)
    
#     elif kg.format.value == 'json':
#         with open(kg.path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             if isinstance(data, list):
#                 for item in data:
#                     if isinstance(item, dict):
#                         if all(key in item for key in ['subject', 'predicate', 'object']):
#                             triple = f"{item['subject']} {item['predicate']} {item['object']}"
#                             triples.add(triple)
    
#     return triples

# # source entity integration

# def evaluate_verified_entity_integration(): pass

#     try:
#         target_entities = self._extract_entities(kg)
#         reference_entities = self._extract_entities(reference_kg)
        
#         intersection = target_entities & reference_entities
#         union = target_entities | reference_entities
        
#         if len(union) == 0:
#             overlap_score = 0.0
#         else:
#             overlap_score = len(intersection) / len(union)
        
#         return MetricResult(
#             name=self.name,
#             value=overlap_score,
#             normalized_score=overlap_score,  # Already 0-1
#             details={
#                 "target_entities": len(target_entities),
#                 "reference_entities": len(reference_entities),
#                 "intersection": len(intersection),
#                 "union": len(union)
#             },
#             aspect=self.aspect
#         )
        
#     except Exception as e:
#         return MetricResult(
#             name=self.name,
#             value=0.0,
#             normalized_score=0.0,
#             details={"error": str(e)},
#             aspect=self.aspect
#         )

#     def _overlap_for_namespace(self, kg: KG, reference_kg: KG, namespace: str) -> Set[str]:
#         """Calculate entity overlap for a specific namespace."""
#         target_entities = self._extract_entities(kg)
#         reference_entities = self._extract_entities(reference_kg)
        
#         intersection = target_entities & reference_entities
#         union = target_entities | reference_entities

#         return intersection
    
#     def _extract_entities(self, kg: KG) -> Set[str]:
#         """Extract entities from KG file."""
#         entities = set()
        
#         g = kg.get_graph()
#         for s, p, o in g:
#             entities.add(str(s))
#             if isinstance(o, URIRef):
#                 entities.add(str(o))
        
#         return entities

# def evaluate_verified_relation_integration_fuzzy_labels(): pass
# def evaluate_verified_relation_integration_with_paris(): pass

# # reference kg

# def evaluate_verified_triple_alignment(): 

#     target_triples = self._extract_triples(kg)
#     reference_triples = self._extract_triples(reference_kg)

#     intersection = target_triples & reference_triples
#     union = target_triples | reference_triples

#     if len(union) == 0:
#         overlap_score = 0.0
#     else:
#         overlap_score = len(intersection) / len(union)

# def evaluate_verified_triple_alignment_fuzzy_literal(): pass
# def evaluate_verified_triple_alignment_with_paris(): pass

# def evaluate_entity_type_derivation(): pass 
        # """
        # based on the reference entity type occurences
        # check the target entity type occurences
        # Example:
        # Reference:
        # - Person: 70
        # - Organization: 55
        # Target:
        # - Person: 100
        # - Organization: 50
        # - Location: 10

        # # TODO requires inference/enrichment of entity types in the kgs
        # """

        # # 1 load reference kg
        # reference_kg = get_test_kg("inc/reference_1.nt")
        # # 2 load target kg
        # target_kg = get_test_kg("inc/rdf_0.nt")

        # occurence_dict_reference = defaultdict(int)
        # reference_entities = get_entities(reference_kg)
        # for entity in reference_entities:
        #     for type in reference_kg.objects(URIRef(entity), RDF.type):
        #         occurence_dict_reference[type] += 1

        # occurence_dict_target = defaultdict(int)
        # target_entities = get_entities(target_kg)
        # for entity in target_entities:
        #     for type in target_kg.objects(URIRef(entity), RDF.type):
        #         occurence_dict_target[type] += 1

        # # normalize distributions
        # def normalize_distribution(counts: dict) -> dict:
        #     total = sum(counts.values())
        #     return {k: v / total for k, v in counts.items()}

        # print(occurence_dict_reference)
        # print(occurence_dict_target)

        # ref_dist = normalize_distribution(occurence_dict_reference)
        # tgt_dist = normalize_distribution(occurence_dict_target)

        # print(ref_dist)
        # print(tgt_dist)

        # # align distributions
        # def align_distributions(ref_dist, tgt_dist):
        #     all_keys = set(ref_dist) | set(tgt_dist)
        #     return (
        #         {k: ref_dist.get(k, 0.0) for k in all_keys},
        #         {k: tgt_dist.get(k, 0.0) for k in all_keys}
        #     )

        # ref_dist, tgt_dist = align_distributions(ref_dist, tgt_dist)

        # print(ref_dist)
        # print(tgt_dist)

        # # calculate distance
        # def calculate_distance(ref_dist, tgt_dist):
        #     # L1 Distance (Manhattan)
        #     l1_distance = sum(abs(ref_dist[k] - tgt_dist[k]) for k in ref_dist)

        #     # L2 Distance (Euclidean)
        #     l2_distance = sum((ref_dist[k] - tgt_dist[k])**2 for k in ref_dist)

        #     return l1_distance, l2_distance

        # l1_distance, l2_distance = calculate_distance(ref_dist, tgt_dist)

        # # KL Divergence (if you want directional)
        # # TODO: check division by zero
        # # TODO: check correct implementation
        # kl_divergence = sum(ref_dist[k] * math.log(ref_dist[k] / tgt_dist[k]) for k in ref_dist if tgt_dist[k] != 0)

        # # Cosine Similarity (distribution similarity)
        # cosine_similarity = sum(ref_dist[k] * tgt_dist[k] for k in ref_dist)

        # print(f"L1 Distance: {l1_distance}")
        # print(f"L2 Distance: {l2_distance}")
        # print(f"KL Divergence: {kl_divergence}")
        # print(f"Cosine Similarity: {cosine_similarity}")

# TODO: schema overlap
