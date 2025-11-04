from pathlib import Path
from dataclasses import dataclass
from typing import Dict
import json
from kgpipe.common.models import KG, KgPipePlan, DataFormat
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document, TE_Pair

from kgpipe_tasks.common.benchutils import hash_uri
from kgpipe.datasets.multipart_multisource import read_links_csv

@dataclass
class LinkCounts:
    true_link_cnt: int
    false_link_cnt: int # can not be derived from the expected links as it is not complete (or if it is an unexpected film)
    true_missing_link_cnt: int # can not be derived from the expected links as it is not complete
    false_missing_link_cnt: int


def load_expected_links_from_json(expected_links_path: Path) -> Dict[str, list[str]]:
    return json.load(open(expected_links_path, "r"))

def load_expected_links_from_csv(expected_links_path: Path) -> Dict[str, list[str]]:
    return{}

def load_expected_entity_links(expected_entites_path: Path):
    """
    {
       "doc_id_1": [
        "entity_id_1",
        "entity_id_2",
        "entity_id_3"
       ],
       "doc_id_2": []
    }
    """
    if expected_entites_path.name.endswith(".json"):
        return load_expected_links_from_json(expected_entites_path)
    elif expected_entites_path.name.endswith(".csv"):
        return load_expected_links_from_csv(expected_entites_path)
    else:
        raise ValueError(f"Unsupported file type: {expected_entites_path.name}")

def get_te_doc_paths_from_plan(plan: KgPipePlan, base_dir) -> list[Path]:
    paths = []
    for step in plan.steps:
        # print(step.output[0])
        if step.output[0].format.value == DataFormat.TE_JSON.value:
            paths.append(step.output[0].path)
    return paths


def get_as_seed_uri(uri: str) -> str:
    hash = hash_uri(uri)
    return f"http://kg.org/resource/{hash}"

def check_links(links: list[TE_Pair], expected_entity_ids: list[str], link_type: str, threshold: float = 0.5):
    true_link_cnt = 0

    for link in links:
        # print(link, expected_entity_ids)
        if link.link_type == link_type and link.score > threshold:
            if link.mapping in expected_entity_ids or link.mapping in [get_as_seed_uri(e) for e in expected_entity_ids]:
                true_link_cnt += 1
            # else:
            #     print(f"False link: {link.mapping} not in expected entity ids {expected_entity_ids} and as seed uri {[get_as_seed_uri(e) for e in expected_entity_ids]}")

    false_missing_link_cnt = len(expected_entity_ids) - true_link_cnt

    return true_link_cnt, false_missing_link_cnt


def __evaluate_expected_links(expected_entites_path: Path, kg: KG, link_type: str, threshold: float = 0.5):

    true_link_cnt = 0
    false_missing_link_cnt = 0

    plan = kg.plan
    if plan is None:
        raise ValueError("KG has no plan")

    expected_entity_links = {row.doc_id.split(".")[0]: [row.entity_id] for row in read_links_csv(expected_entites_path)} #load_expected_entity_links(expected_entites_path)

    te_doc_paths = get_te_doc_paths_from_plan(plan, base_dir=kg.path.parent)
    # print("te_doc_paths", te_doc_paths)
    if len(te_doc_paths) == 0:
        raise ValueError("KG has no te_doc_paths")

    te_doc_path = te_doc_paths[-1]

    if te_doc_path.is_dir():
        for file in te_doc_path.iterdir():
            # print(file)
            doc_id = file.name.split(".")[0]
            te_doc = TE_Document(**json.load(open(file, "r")))

            if doc_id in expected_entity_links:
                expected_entity_ids = expected_entity_links[doc_id]
                doc_true_link_cnt, doc_false_missing_link_cnt = check_links(te_doc.links, expected_entity_ids, link_type, threshold)

                true_link_cnt += doc_true_link_cnt
                false_missing_link_cnt += doc_false_missing_link_cnt
            else:
                print(f"Doc {doc_id} not in expected_entity_links")
                pass
    else:
        # print(f"te_doc_path is not a directory: {te_doc_path}")
        pass

    return LinkCounts(
        true_link_cnt=true_link_cnt,
        false_link_cnt=-1,
        true_missing_link_cnt=-1,
        false_missing_link_cnt=false_missing_link_cnt
    )

def evaluate_expected_entity_links(expected_entites_path: Path, kg: KG, threshold: float = 0.5):
    return __evaluate_expected_links(expected_entites_path, kg, "entity", threshold)

def evaluate_expected_relation_links(expected_relations_path: Path, kg: KG, threshold: float = 0.5):
    return __evaluate_expected_links(expected_relations_path, kg, "relation", threshold)
