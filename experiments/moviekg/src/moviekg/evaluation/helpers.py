from kgpipe.common.models import KG, DataFormat, Data, Metric
from kgpipe.evaluation.aspects import statistical

from kgpipe.evaluation.aspects import reference, semantic, statistical
from kgpipe.common.models import KG, DataFormat
from kgpipe.evaluation.base import MetricResult
from kgpipe.evaluation.writer import render_as_table_multi, render_metric_as_table

from moviekg.datasets.pipe_out import StageOut

from pyodibel.datasets.mp_mf.multipart_multisource import load_dataset, Dataset
from kgcore.model.ontology import  Ontology, OntologyUtil

from typing import List, Dict, Tuple
from pathlib import Path
from rdflib import Graph
import json

from moviekg.config import dataset


ontology_graph = Graph()
if dataset.ontology is None:
    raise ValueError("No ontology found")
ontology_graph.parse(dataset.ontology.as_posix())

def show_ontology():
    if dataset.ontology is None:
        raise ValueError("No ontology found")
    ontology = OntologyUtil.load_ontology_from_file(dataset.ontology)

    for class_ in ontology.classes:
        print(f"{class_.uri} {class_.label}")
        # print(f"{class_.alias} {class_.description}")
        print(f"{class_.equivalent}")
        print(f"{class_.disjointWith}")
        print("-" * 100)
    
    for property in ontology.properties:
        print(f"{property.uri} {property.type} {property.label}")
        # print(f"{property.alias} {property.description}")
        print(f"{property.domain.uri} {property.range.uri} {property.equivalent}")
        print(f"{property.min_cardinality} {property.max_cardinality}")
        print("-" * 100)

show_ontology()


def print_long_table_rows(rows: List[dict]):
    """
    with correct margin and alignment
    """
    max_aspect_length = max(len(row["aspect"]) for row in rows)
    max_metric_name_length = max(len(row["metric"]) for row in rows)
    max_value_length = max(len(str(row["value"])) for row in rows)
    max_normalized_length = max(len(str(row["normalized"])) for row in rows)
    max_duration_length = max(len(str(row["duration"])) for row in rows)

    print(f"{'Aspect':<{max_aspect_length}} | {'Metric':<{max_metric_name_length}} | {'Value':<{max_value_length}} | {'Normalized':<{max_normalized_length}} | {'Duration':<{max_duration_length}}")
    print("-" * (max_aspect_length + max_metric_name_length + max_value_length + max_normalized_length + max_duration_length + 6))
    for row in rows:
        print(f"{row['aspect']:<{max_aspect_length}} | {row['metric']:<{max_metric_name_length}} | {row['value']:<{max_value_length}} | {row['normalized']:<{max_normalized_length}} | {row['duration']:<{max_duration_length}}")

def metrics_to_long_table_rows(metrics: List[MetricResult], pipeline_name: str, stage_name: str) -> List[dict]:
    rows = []
    for metric in metrics:
        rows.append({
            "pipeline": pipeline_name,
            "stage": stage_name,
            "aspect": metric.aspect.value,
            "metric": metric.name,
            "value": metric.value,
            "normalized": metric.normalized_score,
            "duration": metric.duration,
            "details": json.dumps(metric.details, default=str)
        })
    return rows

from kgpipe.evaluation.aspects.reference import ReferenceConfig

def get_reference_config(stage: StageOut, is_ssp: bool) -> ReferenceConfig:

    # this is a pipeline name based hack to get the source type and source split id
    def get_split_id_and_source_type(stage: StageOut, is_ssp: bool = False) -> Tuple[int, str]:
        # stage_name is like "stage_1"
        split_id = int(stage.stage_name.split("_")[1])
        pipeline_name = stage.root.parent.name
        source_ord = pipeline_name.split("_")

        if len(source_ord) != 3 or is_ssp:
            source_type = source_ord[0]
        else:
            source_type = source_ord[split_id-1]
        return split_id, source_type

    split_id, source_type = get_split_id_and_source_type(stage)

    meta = dataset.splits[f"split_{split_id}"].sources[source_type].meta
    verified_source_entities_path = dataset.splits[f"split_{split_id}"].kg_seed.root / "meta/verified_entities.csv"
    verified_source_matches_path = meta.root / "verified_matches.csv"


    kg_reference = dataset.splits[f"split_{split_id}"].kg_reference
    if kg_reference is None:
        raise ValueError(f"No reference KG found for split {split_id} and source type {source_type}")
    reference_path = kg_reference.root / "data_agg.nt"

    kg_seed = dataset.splits[f"split_0"].kg_seed
    if kg_seed is None:
        raise ValueError(f"No seed KG found for split {0}")
    seed_path = kg_seed.root / "data.nt"

    ENTITY_MATCH_THRESHOLD_MAP = {
        "json_baseA": 0.99,
        "json_a": 0.99,
        "rdf_a": 0.99,
        "rdf_b": 0.5,
        "rdf_b2": 0.5,
        "rdf_c": 0.5,
    }

    RELATION_MATCH_THRESHOLD_MAP = {
        "json_a": 0.5,
        "json_baseA": 0.5,
        "rdf_a": 0.5,
        "rdf_b": 0.1,
        "rdf_b2": 0.1,
        "rdf_c": 0.5,
    }

    return ReferenceConfig(
        name="reference",
        GT_MATCHES=verified_source_matches_path,
        GT_MATCHES_TARGET_DATASET=dataset.splits[f"split_{0}"].root.name+"/kg/seed",
        RELATION_MATCH_THRESHOLD=RELATION_MATCH_THRESHOLD_MAP.get(stage.root.parent.name, 0.5),
        ENTITY_MATCH_THRESHOLD=ENTITY_MATCH_THRESHOLD_MAP.get(stage.root.parent.name, 0.5),
        VERIFIED_SOURCE_ENTITIES=verified_source_entities_path,
        REFERENCE_KG_PATH=reference_path,
        SEED_KG_PATH=seed_path,
        # EXPECTED_TEXT_LINKS=Path("TODO"),
        TE_LINK_THRESHOLD=0.5,
        source_meta=meta,
        dataset=dataset,
        JSON_EXPECTED_DIR="/home/marvin/project/data/work/json", #TODO later
        JSON_EXPECTED_RELATION_FILE="/home/marvin/project/data/final/film_10k/split_0/sources/json/meta/verified_relation_matches.json" # TODO later
        #dataset.splits[f"split_{split_id}"].sources["json"].data.dir.as_posix()
    )

from kgpipe.evaluation.base import MetricResult, EvaluationAspect

def add_duration_metrics(stage: StageOut) -> MetricResult:
    
    try:
        duration = stage.report.duration
        return MetricResult(
            aspect=EvaluationAspect.STATISTICAL,
            name="duration",
            value=duration,
            normalized_score=0,
            details={
                "duration": duration
            }
        )
    
    except Exception as e:
        return MetricResult(
            aspect=EvaluationAspect.STATISTICAL,
            name="duration",
            value=0,
            normalized_score=0,
            details={"error": "No duration found"}
        )



def evaluate_stage(stage: StageOut, is_ssp: bool) -> List[MetricResult]:
    result_path = stage.resultKG
    if result_path is None:
        return []

    result_kg = KG(id=f"result_{stage.stage_name}", name=f"result_{stage.stage_name}", path=result_path, format=DataFormat.RDF_NTRIPLES,plan=stage.plan)

    result_kg.set_ontology_graph(ontology_graph)

    stat_eval = statistical.StatisticalEvaluator()
    ref_eval = reference.ReferenceEvaluator()
    sem_eval = semantic.SemanticEvaluator()

    stats_aspect_result = stat_eval.evaluate(result_kg)
    ref_aspect_result = ref_eval.evaluate(result_kg, config=get_reference_config(stage, is_ssp))
    sem_aspect_result = sem_eval.evaluate(result_kg)
    
    metrics = []
    metrics = stats_aspect_result.metrics + ref_aspect_result.metrics + sem_aspect_result.metrics
    # metrics = sem_aspect_result.metrics
    metrics.append(add_duration_metrics(stage))
    # metrics = ref_aspect_result.metrics

    return metrics

import tempfile
import re
import shutil

def replace_with_dict(infile: str, mapping: dict[str, str]) -> None:
    with open(infile, encoding="utf-8") as f, \
         tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        for line in f:
            for key, val in mapping.items():
                line = re.sub(re.escape(key), val, line)
            tmp.write(line)
        tmp_path = tmp.name
    shutil.move(tmp_path, infile)
