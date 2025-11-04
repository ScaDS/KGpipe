from typing import Dict
from pathlib import Path
from kgpipe.common.models import KG, KgPipePlan, DataFormat
from kgpipe.datasets.multipart_multisource import MatchesRow
from rdflib import Graph, URIRef
from typing import Optional
import json
from kgpipe.evaluation.cluster import MatchCluster, is_match, load_matches
from dataclasses import dataclass
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document, ER_Match

# utility
def load_kg(path: Path | list[Path]) -> Graph:
    if isinstance(path, Path):
        kg = Graph()
        kg.parse(path.as_posix(), format="turtle")
        return kg
    else:
        kg = Graph()
        for p in path:
            kg.parse(p.as_posix(), format="turtle")
        return kg

def load_entities(path: Path | list[Path]) -> set[str]:
    """
    loads entity list from .txt file
    """
    if isinstance(path, Path):
        with open(path, "r") as f:
            return set([line.strip() for line in f.readlines()])
    else:
        return set.union(*[load_entities(p) for p in path])

def load_cluster_from_em_json(cluster: MatchCluster, em_json: Path, threshold: float = 0.5):
    """
    loads cluster from em_json file
    """
    data: ER_Document= ER_Document(**json.load(open(em_json, "r")))
    matches: list[ER_Match] = data.matches
    for match in matches:
        if match.score > threshold:
            cluster.add_match(match.id_1, match.id_2)
    return cluster


# get entities
def get_entities(kg: Graph) -> set[str]:
    return set([str(s) for s in kg.subjects()]+[str(o) for o in kg.objects() if isinstance(o, URIRef)])
# get relations
def get_relations(kg: Graph) -> set[str]:
    return set([str(p) for p in kg.predicates()])
# get triples
def get_triples(kg: Graph) -> set[str]:
    return set([str(t) for t in kg.triples((None, None, None))])

@dataclass
class MatchCounts:
    true_entity_match_cnt: int
    false_entity_match_cnt: int
    true_missing_entity_match_cnt: int
    false_missing_entity_match_cnt: int
    true_relation_match_cnt: int
    false_relation_match_cnt: int
    true_missing_relation_match_cnt: int
    false_missing_relation_match_cnt: int


def get_matches(er_doc: ER_Document, match_cluster: Optional[MatchCluster] = None, allow_match_on_suffix: bool = True):
    true_entity_match_cnt = 0
    false_entity_match_cnt = 0
    true_missing_entity_match_cnt = 0
    false_missing_entity_match_cnt = 0
    true_relation_match_cnt = 0
    false_relation_match_cnt = 0
    true_missing_relation_match_cnt = 0
    false_missing_relation_match_cnt = 0

    check_matches = set()

    for match in er_doc.matches:
        score = match.score
        id_1 = str(match.id_1)
        id_2 = str(match.id_2)
        if not score:
            score = 0
        if score > 0.5 and match.id_type == "entity":
            if is_match(id_1, id_2, match_cluster, allow_match_on_suffix):
                # print(f"True match: {match.id_1} {match.id_2}")
                true_entity_match_cnt += 1
            else:
                false_entity_match_cnt += 1
        elif score > 0.5 and match.id_type == "relation":
            id_1 = str(match.id_1) # .split("-")[0] # paris workaround
            id_2 = str(match.id_2) #.split("-")[0] # paris workaround
            if id_1.endswith("-") or id_2.endswith("-"):
                # print(f"Skipping match: {id_1} {id_2}")
                continue
            if id_1 in check_matches and id_2 in check_matches:
                # print(f"Skipping match: {id_1} {id_2}")
                continue
            check_matches.add(id_1)
            check_matches.add(id_2)
            if is_match(id_1, id_2, match_cluster, allow_match_on_suffix):
                # print(f"True match: {match.id_1} {match.id_2}")
                true_relation_match_cnt += 1
            else:
                # print(f"False match: score {score} {id_1} {id_2}")
                false_relation_match_cnt += 1
        else:
            if match.id_type == "entity":
                true_missing_entity_match_cnt += 1
            else:
                true_missing_relation_match_cnt += 1

    return MatchCounts(
        true_entity_match_cnt=true_entity_match_cnt,
        false_entity_match_cnt=false_entity_match_cnt,
        true_missing_entity_match_cnt=true_missing_entity_match_cnt,
        false_missing_entity_match_cnt=false_missing_entity_match_cnt,
        true_relation_match_cnt=true_relation_match_cnt,
        false_relation_match_cnt=false_relation_match_cnt,
        true_missing_relation_match_cnt=true_missing_relation_match_cnt,
        false_missing_relation_match_cnt=false_missing_relation_match_cnt
    )

def get_relation_matches(er_doc: ER_Document, threshold: float, match_cluster: Optional[MatchCluster] = None, allow_match_on_suffix: bool = True):
    true_relation_match_cnt = 0
    false_relation_match_cnt = 0

    check_matches = set()

    for match in er_doc.matches:
        score = match.score
        id_1 = str(match.id_1)
        id_2 = str(match.id_2)
        if not score:
            score = 0
        if score > threshold and match.id_type == "relation":
            id_1 = str(match.id_1) # .split("-")[0] # paris workaround
            id_2 = str(match.id_2) #.split("-")[0] # paris workaround
            if id_1.endswith("-") or id_2.endswith("-"):
                # print(f"Skipping match: {id_1} {id_2}")
                continue
            if id_1 in check_matches and id_2 in check_matches:
                # print(f"Skipping match: {id_1} {id_2}")
                continue
            check_matches.add(id_1)
            check_matches.add(id_2)
            if is_match(id_1, id_2, match_cluster, allow_match_on_suffix):
                # print(f"True match: {match.id_1} {match.id_2}")
                true_relation_match_cnt += 1
            else:
                # print(f"False match: score {score} {id_1} {id_2}")
                false_relation_match_cnt += 1

    return MatchCounts(
        true_entity_match_cnt=0,
        false_entity_match_cnt=0,
        true_missing_entity_match_cnt=0,
        false_missing_entity_match_cnt=0,

        true_relation_match_cnt=true_relation_match_cnt,
        false_relation_match_cnt=false_relation_match_cnt,
        true_missing_relation_match_cnt=0,
        false_missing_relation_match_cnt=23 - (true_relation_match_cnt + false_relation_match_cnt) # TODO add this workaround for now
    )

def get_matches_to_seed(er_doc: ER_Document, list_of_matches: list[MatchesRow], threshold):

    print("list of matches", len(list_of_matches))
    print("threshold", threshold)

    true_entity_match_cnt = 0 #tp
    false_entity_match_cnt = 0 #fp
    false_missing_entity_match_cnt = 0 #fn

    gt_cluster = MatchCluster()
    gt_seed_ids = set()
    for match in list_of_matches:
        gt_seed_ids.add(match.right_id)
        gt_cluster.add_match(match.left_id, match.right_id)

    actual_cluster = MatchCluster()

    for match in er_doc.matches:
        if match.score > threshold:
            # print("adding match", match.id_1, match.id_2)
            if match.id_1 != match.id_2:
                actual_cluster.add_match(match.id_1, match.id_2)

    saw_seed_ids = set()

    for cid, actual_match in actual_cluster.clusters.items():
        match_list = list(actual_match)
        id1 = match_list[0]
        id2 = match_list[1]

        # get the seed and source ids
        seed_id = None
        source_id = None
        if id1.startswith("http://kg.org/resource"):
            source_id = id2
            seed_id = id1
        if id2.startswith("http://kg.org/resource"):
            source_id = id1
            seed_id = id2

        checker = False
        if seed_id == "http://kg.org/resource/b25598f9c0fce28a7700869fcb55d706":
            checker = True

   
        if seed_id is not None and source_id is not None:
            saw_seed_ids.add(seed_id)
            if is_match(source_id, seed_id, gt_cluster, False):
                true_entity_match_cnt += 1
                if checker:
                    print("true match", seed_id, source_id)
            else:
                false_entity_match_cnt += 1
                if checker:
                    print("false match", seed_id, source_id)
        else:
            # can not be checked, skip
            if checker:
                print("skip", seed_id, source_id)
            continue

    missing_seed_ids = gt_seed_ids - saw_seed_ids
    false_missing_entity_match_cnt = len(missing_seed_ids)

    return MatchCounts(
        true_entity_match_cnt=true_entity_match_cnt,
        false_entity_match_cnt=false_entity_match_cnt,
        true_missing_entity_match_cnt=0,
        false_missing_entity_match_cnt=false_missing_entity_match_cnt,  

        # TODO add relation matching
        true_relation_match_cnt=0,
        false_relation_match_cnt=0,
        true_missing_relation_match_cnt=0,
        false_missing_relation_match_cnt=0
    )
                

    # for match in list_of_matches:
    #     left_cluster = actual_cluster.get_cluster(match.left_id)
    #     right_cluster = actual_cluster.get_cluster(match.right_id)

    #     if left_cluster is None or right_cluster is None:
    #         false_missing_entity_match_cnt += 1
    #     elif is_match(match.left_id, match.right_id, actual_cluster, False):
    #         true_entity_match_cnt += 1
    #     else:
    # false_entity_match_cnt += 1


def check_matches(matches: Dict[str, str], match_cluster: Optional[MatchCluster] = None, allow_match_on_suffix: bool = False):
    true_match_cnt = 0
    false_match_cnt = 0
    
    for span, mapping in matches.items():
        if is_match(span, mapping, match_cluster, allow_match_on_suffix):
            true_match_cnt += 1
        else:
            # print(f"False match: {span} {mapping}")
            # print(f"Expected: {match_cluster.get_cluster(span) if match_cluster else None}")
            # print(f"Actual: {match_cluster.get_cluster(mapping) if match_cluster else None}")
            false_match_cnt += 1
    return true_match_cnt, false_match_cnt


# """
# checks ground truth matches agains actual matches
# TODO find exisiting implementation in codebase
# """

def get_er_doc_paths_from_plan(plan: KgPipePlan, base_dir) -> list[Path]:
    paths = []
    for step in plan.steps:
        # print(step.output[0])
        if step.output[0].format.value == DataFormat.ER_JSON.value:
            paths.append(step.output[0].path)
    return paths

def get_er_doc_from_kg(kg: KG) -> ER_Document:
    plan = kg.plan
    if plan is None:
        raise ValueError("KG has no plan")
    paths = get_er_doc_paths_from_plan(plan, base_dir=kg.path.parent)
    # print("paths", paths)
    if len(paths) == 0:
        raise ValueError("KG has no er_doc_paths")
    # print("loading er_doc from", paths[-1])
    return ER_Document(**json.load(open(paths[-1]))) # only the last one

def load_actual_matches(er_doc_path: Path | KG, threshold: float = 0.5, type_filter: str = "") -> MatchCluster:
    if isinstance(er_doc_path, KG):
        er_doc = get_er_doc_from_kg(er_doc_path)
    else:
        er_doc = ER_Document(**json.load(open(er_doc_path, "r")))
    
    # BIG TODO: this needs to be read correctly as there are inverse matches aswell
    cluster = MatchCluster()
    for match in er_doc.matches:
        if match.score > threshold and (match.id_type == type_filter or type_filter == ""):
            if match.id_1.endswith("-") or match.id_2.endswith("-"):
                continue
            cluster.add_match(match.id_1, match.id_2)
    return cluster

def load_expected_matches_from_json(gt_match_path: Path) -> MatchCluster:
    data: ER_Document= ER_Document(**json.load(open(gt_match_path, "r")))
    matches: list[ER_Match] = data.matches
    cluster = MatchCluster()
    for match in matches:
        cluster.add_match(match.id_1, match.id_2)
    return cluster

import csv

def load_expected_matches_from_csv(gt_match_path: Path, gt_match_target_dataset: str) -> MatchCluster:
    
    match_rows = []
    with open(gt_match_path, "r") as f:
        reader = csv.reader(f,delimiter="\t")
        for row in reader:
            match_rows.append(MatchesRow(left_dataset=row[0], right_dataset=row[1], left_id=row[2], right_id=row[3], entity_type=row[4]))
    
    match_cluster = MatchCluster()
    for match_row in match_rows:
        # print(match_row.left_dataset, match_row.right_dataset)
        # print(gt_match_target_dataset)
        if match_row.left_dataset == gt_match_target_dataset or match_row.right_dataset == gt_match_target_dataset:
            match_cluster.add_match(match_row.left_id, match_row.right_id)
    return match_cluster

def load_expected_matches(gt_match_path: Path, gt_match_target_dataset: str) -> MatchCluster:
    if gt_match_path.name.endswith(".json"):
        return load_expected_matches_from_json(gt_match_path)
    elif gt_match_path.name.endswith(".csv"):
        return load_expected_matches_from_csv(gt_match_path, gt_match_target_dataset)
    else:
        raise ValueError(f"Unsupported file extension: {gt_match_path.name}")

def load_matches_tuples_to_target(gt_match_path: Path, gt_match_target_dataset: str) -> list[MatchesRow]:
    match_rows = []
    with open(gt_match_path, "r") as f:
        reader = csv.reader(f,delimiter="\t")
        for row in reader:
            if gt_match_target_dataset in row[0] or gt_match_target_dataset in row[1]:
                match_rows.append(MatchesRow(left_dataset=row[0], right_dataset=row[1], left_id=row[2], right_id=row[3], entity_type=row[4]))
    
    return match_rows
# public functions

# TODO: 
# tp: correct match (entity that has match, should have match, and is correct)
# fp: incorrect match (entity that has match but is not correct, and entity that has match but should not have)
# fn: missing match (entity that should have match but does not)
# # Correct Matches (TP)
# # Incorrect Matches (FP)
# # Correct No Matches (TN)
# # Incorrect No Matches (FN)

# TODO false negative needs to be derived from the gt matches
# TODO this is not finished
def evaluate_entity_matching(er_doc_path_or_kg: Path | KG, 
                             gt_match_path: Path, 
                             gt_match_target_dataset: str,
                             threshold) -> tuple[float, float, dict]: 
    
    gt_matches = load_matches_tuples_to_target(gt_match_path, gt_match_target_dataset)

    # print(f"SIZE OF GT MATCHES to {gt_match_target_dataset} is {len(gt_matches)}")
    # # print(gt_matches)

    if isinstance(er_doc_path_or_kg, KG):
        er_doc = get_er_doc_from_kg(er_doc_path_or_kg)
    else:
        er_doc = ER_Document(**json.load(open(er_doc_path_or_kg, "r")))
    
    match_counts = get_matches_to_seed(er_doc, gt_matches, threshold)
    tp = match_counts.true_entity_match_cnt
    fp = match_counts.false_entity_match_cnt
    fn = match_counts.false_missing_entity_match_cnt

    denom = 2 * tp + fp + fn
    f1_score = (2 * tp / denom) if denom > 0 else 0.0

    print("f1_score", f1_score)
    print("tp", tp)
    print("fp", fp)
    print("fn", fn)

    return f1_score, f1_score, {"true_seed_match_cnt": tp, "false_seed_match_cnt": fp, "false_missing_seed_match_cnt": fn}

# TODO not finished
def evaluate_relation_matching(er_doc_path: Path | KG, gt_match_path: Path, threshold: float) -> tuple[float, float, dict]:
    
    if isinstance(er_doc_path, KG):
        er_doc = get_er_doc_from_kg(er_doc_path)
    else:
        er_doc = ER_Document(**json.load(open(er_doc_path, "r")))
    
    match_counts = get_relation_matches(er_doc, threshold, None, True)
    tp = match_counts.true_relation_match_cnt
    fp = match_counts.false_relation_match_cnt
    fn = match_counts.false_missing_relation_match_cnt
    
    f1_score = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0
    
    return f1_score, f1_score, {"true_relation_match_cnt": tp, "false_relation_match_cnt": fp, "false_missing_relation_match_cnt": fn}
