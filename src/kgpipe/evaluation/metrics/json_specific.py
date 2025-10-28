
"""
Evaluation of entity resolution from JSON-to-KG mapping.
"""

from curses import tparm
import os
import json
from typing import Mapping, Dict, List, Tuple
from kgpipe.evaluation.base import Metric, MetricResult, MetricConfig
from kgpipe.common.models import KG
from kgpipe.evaluation.cluster import MatchCluster
from kgpipe.evaluation.aspects.func.integration_eval import BinaryClassificationResult
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document
from kgpipe.common.models import KgPipePlan
from kgpipe.common.models import DataFormat
from pathlib import Path
from typing import Optional
from kgpipe.common.registry import Registry
from pyodibel.common.util import hash_uri

class JsonSpecificConfig(MetricConfig):
    JSON_EXPECTED_DIR: Optional[str] = None

def get_json_mapping_prov(kg: KG,) -> list[Path]:
    plan = kg.plan
    if plan is None:
        raise ValueError("KG has no plan")
    paths = []
    for step in plan.steps:
        # print(step.output[0])
        if "construct_rdf_from_json2" in step.output[0].path.name or "construct_linkedrdf_from_json_v2" in step.output[0].path.name:
            paths.append(Path(step.output[0].path.as_posix()+".prov"))
    return paths

def get_er_task_json_doc(kg: KG) -> list[Path]:
    plan = kg.plan
    if plan is None:
        raise ValueError("KG has no plan")
    paths = []
    for step in plan.steps:
        # print(step.output[0])
        if step.output[0].format.value == DataFormat.ER_JSON.value:
            paths.append(Path(step.output[0].path.as_posix()))
    return paths

"""
get expected mapping
get actual mapping
"""
def evaluate_json_a_matching(JSON_EXPECTED_DIR: str, JSON_ACTUAL_FILE: str, MATCH_JSON_FILE: str, MATCH_THRESHOLD: float):


    def get_expected_mapping():
        file_mappings: Dict[str, Mapping[str, str]] = {}
        for file in os.listdir(JSON_EXPECTED_DIR):
            if file.endswith(".prov"):
                with open(os.path.join(JSON_EXPECTED_DIR, file), "r") as f:
                    data: Mapping[str, str]  = json.load(f)
                    file_mappings[file.replace(".prov", ".json")] = data
        return file_mappings


    def get_actual_mapping():
        with open(JSON_ACTUAL_FILE, "r") as f:
            data: Dict[str, Mapping[str, str]]  = json.load(f)
            return data


    def get_actual_matches():
        from kgpipe.evaluation.aspects.func.er_task_eval import load_actual_matches
        return load_actual_matches(MATCH_JSON_FILE, MATCH_THRESHOLD, type_filter="entity")

    def generate_pairs_from_clusters(clusters: MatchCluster) -> List[Tuple[str, str]]:
        pairs = []
        for cluster in clusters.clusters.values():
            pair = list[str](cluster) # type: ignore
            if len(pair) > 2:
                # print(cluster)
                continue # TODO
                raise ValueError(f"Cluster with more than 2 elements: {pair} ... NOT SUPPORTED")
            pair.sort()
            # TODO create all slices of the pair to avoid len > 2 clusters
            pairs.append(tuple[str, str](pair))
        return pairs
    
    def get_expected_matches(expected_mapping: Dict[str, Mapping[str, str]], actual_mapping: Dict[str, Mapping[str, str]]):
        # based on expected mapping and actual mapping
        cluster = MatchCluster()
        for filename, _path_mappings in actual_mapping.items(): # TODO issue not correct 100,1k,10k 
            # print(filename)
            path_mappings: Mapping[str, str] = _path_mappings
            if filename in expected_mapping:
                for json_path in path_mappings:
                    if json_path in expected_mapping[filename]:
                        cluster.add_match("http://kg.org/resource/" + hash_uri(expected_mapping[filename][json_path]), path_mappings[json_path])
                    else:
                        # print(f"Path {json_path} not found in expected mapping")
                        continue
            else:
                # print(f"Filename {filename} not found in actual mapping")
                continue
        return cluster

    """
    True Positive (TP) = number of pairs that appear in both expected and predicted sets.
    False Positive (FP) = number of pairs that appear in predicted but not in expected.
    False Negative (FN) = number of pairs that appear in expected but not in predicted.
    """

    expected_mapping = get_expected_mapping()
    actual_mapping = get_actual_mapping()
    actual_matches = get_actual_matches()
    expected_matches = get_expected_matches(expected_mapping, actual_mapping)

    expected_pairs = set(generate_pairs_from_clusters(expected_matches))
    actual_pairs = set(generate_pairs_from_clusters(actual_matches))



    tp_pairs = expected_pairs & actual_pairs
    fp_pairs = actual_pairs - expected_pairs
    fn_pairs = expected_pairs - actual_pairs

    tp = len(tp_pairs)
    fp = len(fp_pairs)
    fn = len(fn_pairs)

    print("debug --------------------------------")
    print("expected_pairs", len(expected_pairs))
    print("actual_pairs", len(actual_pairs))
    print("expected_mapping", len(expected_mapping))
    print("actual_mapping", len(actual_mapping))
    print("fp", fp)
    print("fn", fn)
    print("tp", tp)
    print("debug --------------------------------")


    binary_classification_metrics = BinaryClassificationResult(tp, fp, 0, fn)
    return binary_classification_metrics

def evaluate_json_b_linking(JSON_EXPECTED_DIR: str, JSON_ACTUAL_FILE: str):
    """
    Evaluate the linking of JSON-to-KG mapping.
    """

    def get_expected_mapping():
        file_mappings: Dict[str, Mapping[str, str]] = {}
        for file in os.listdir(JSON_EXPECTED_DIR):
            if file.endswith(".prov"):
                with open(os.path.join(JSON_EXPECTED_DIR, file), "r") as f:
                    data: Mapping[str, str]  = json.load(f)
                    file_mappings[file.replace(".prov", ".json")] = data
        return file_mappings

    def get_actual_mapping():
        with open(JSON_ACTUAL_FILE, "r") as f:
            data: Dict[str, Mapping[str, str]]  = json.load(f)
            return data

    expected_mapping = get_expected_mapping()
    actual_mapping = get_actual_mapping()

    tp = 0
    fp = 0
    fn = 0

    # print(list(expected_mapping.items())[:2])   
    # print(list(actual_mapping.items())[:2])

    for filename, _path_mappings in actual_mapping.items(): # TODO issue not correct 100,1k,10k 
        # print(filename)
        path_mappings: Mapping[str, str] = _path_mappings
        if filename in expected_mapping:
            # compare file mappings 
            for json_path in path_mappings:
                if json_path in expected_mapping[filename]:
                    actual_link = path_mappings[json_path]
                    expected_link = "http://kg.org/resource/" + hash_uri(expected_mapping[filename][json_path])
                    if actual_link == expected_link:
                        tp += 1
                    else:
                        if actual_link.startswith("http://kg.org/json/"):
                            fn += 1
                        else:
                            fp += 1
                else:
                    if actual_link.startswith("http://kg.org/json/"):
                        pass # this is expected
                    else:
                        fp += 1
        else:
            # mapped file but not expected
            # TODO maybe raise error
            pass

    binary_classification_metrics = BinaryClassificationResult(tp, fp, 0, fn)
    return binary_classification_metrics

if __name__ == "__main__":

    # TODO check if correct for increment eval if namespace changes to former generic namespace not seed

    B_JSON_EXPECTED_DIR="/home/marvin/project/data/work/json"
    B_JSON_ACTUAL_FILE="/home/marvin/project/data/out/medium/json_b/stage_1/tmp/0_construct_linkedrdf_from_json_v2_0.nt.prov"

    evaluate_json_b_linking(B_JSON_EXPECTED_DIR, B_JSON_ACTUAL_FILE)

    A_JSON_EXPECTED_DIR="/home/marvin/project/data/work/json"
    A_JSON_ACTUAL_FILE="/home/marvin/project/data/out/medium/json_a/stage_1/tmp/0_construct_rdf_from_json2_0.nt.prov"
    A_MATCH_THRESHOLD = 0.5
    A_MATCH_JSON_FILE="/home/marvin/project/data/out/medium/json_a/stage_1/tmp/2_paris_exchange_0.er.json"
    evaluate_json_a_matching(A_JSON_EXPECTED_DIR, A_JSON_ACTUAL_FILE, A_MATCH_JSON_FILE, A_MATCH_THRESHOLD)

