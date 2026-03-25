from dataclasses import dataclass
from typing import List
from kgpipe.common import KgPipe, Data, DataFormat, KG
from pathlib import Path
from kgpipe.common.models import KgPipePlan
from kgpipe.evaluation.aspects.reference import (
    ReferenceEvaluator, ReferenceConfig, 
    ER_EntityMatchMetric, ER_RelationMatchMetric, 
    TE_ExpectedEntityLinkMetric, TE_ExpectedRelationLinkMetric
)
import os
@dataclass
class BinaryClassifier:
    tp: int
    fp: int
    tn: int
    fn: int

    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)

@dataclass
class ThresholdSensitivityResult:
    pipeline_name: str
    threshold: float
    result: BinaryClassifier

benchdata = Path("/home/marvin/phd/kgpipe/experiments/moviekg/data/datasets/film_10k/")
seed_path = benchdata / "split_0/kg/seed/data.nt"
rdf_path = benchdata / "split_1/sources/rdf/data.nt"
result_dir_path = Path(f"data/moviekg/threshold_sensitivity/")

# reference_evaluator = ReferenceEvaluator()

def run_paris_pipeline(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    from kgpipe_tasks.tasks import paris_entity_matching, paris_exchange

    pipe_result_dir_path = result_dir_path / f"{pipeline_name}"
    pipeline = KgPipe(
        name="paris pipeline",
        tasks=[paris_entity_matching, paris_exchange],
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=pipe_result_dir_path / "tmp"
    )
    plan = pipeline.build(
        source=Data(path=rdf_path, format=DataFormat.RDF_NTRIPLES),
        result=Data(path=pipe_result_dir_path / "result.json", format=DataFormat.ER_JSON)
    )

    os.makedirs(pipe_result_dir_path, exist_ok=True)

    with open(pipe_result_dir_path / "exec-plan.json", "w") as f:
        f.write(plan.model_dump_json(indent=4))

    pipeline.run()

def paris_er_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    config = ReferenceConfig(
        name="paris config",
        ENTITY_MATCH_THRESHOLD=threshold,
        RELATION_MATCH_THRESHOLD=threshold,
        GT_MATCHES=benchdata / "split_1/sources/rdf/meta/verified_matches.csv",
        GT_MATCHES_TARGET_DATASET="split_0/kg/seed"
    ) 

    plan = KgPipePlan.model_validate_json(open(result_dir_path / f"{pipeline_name}" / "exec-plan.json").read())

    kg = KG(id="paris", name="paris", path=Path(f"data/moviekg/paris/{pipeline_name}.nt"), format=DataFormat.RDF_NTRIPLES, plan=plan)

    metric_result = ER_EntityMatchMetric().compute(kg, config=config)
    # print(metric_result)

    return metric_result

def paris_om_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    config = ReferenceConfig(
        name="paris config",
        ENTITY_MATCH_THRESHOLD=threshold,
        RELATION_MATCH_THRESHOLD=threshold,
        GT_MATCHES=benchdata / "split_1/sources/rdf/meta/verified_matches.csv",
        GT_MATCHES_TARGET_DATASET="split_0/kg/seed"
    ) 

    plan = KgPipePlan.model_validate_json(open(result_dir_path / f"{pipeline_name}" / "exec-plan.json").read())

    kg = KG(id="paris", name="paris", path=Path(f"data/moviekg/paris/{pipeline_name}.nt"), format=DataFormat.RDF_NTRIPLES, plan=plan)

    metric_result = ER_RelationMatchMetric().compute(kg, config=config)
    # print(metric_result)

    return metric_result

def test_paris():
    # run_paris_pipeline("paris", 0.99)
    range_of_thresholds = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
    
    er_results = []
    for threshold in range_of_thresholds:
        result = paris_er_threshold_sensitivity("paris", threshold)
        er_results.append([threshold, result.normalized_score, result.details])

    print()
    print("ER Results:")
    for r in er_results:
        print(r[0], r[1], r[2])

    om_results = []
    for threshold in range_of_thresholds:
        result = paris_om_threshold_sensitivity("paris", threshold)
        om_results.append([threshold, result.normalized_score, result.details])

    print("OM Results:")
    for r in om_results:
        print(r[0], r[1], r[2])

# def paris_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
#     result = run_paris_pipeline(pipeline_name, threshold)

#     pipeline.run(
#         input=[Data(path=Path(f"data/moviekg/paris/{pipeline_name}.nt"), format=DataFormat.RDF_NTRIPLES)],
#         output=[Data(path=Path(f"data/moviekg/paris/{pipeline_name}.paris_csv"), format=DataFormat.PARIS_CSV)]
#     )

#     config = ReferenceConfig(
#         name="paris config",
#         ENTITY_MATCH_THRESHOLD=threshold,
#         RELATION_MATCH_THRESHOLD=threshold        
#     ) 




#     # TODO get config from dataset
#     # kg = KG(path=Path(f"data/moviekg/paris/{pipeline_name}.nt"))
#     # reference_kg = KG(path=Path("data/moviekg/paris/reference.nt"))
#     # result = reference_evaluator.evaluate(kg, reference_kg)
#     # return result
#     pass

def jedai_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def valentine_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def corenlp_openie_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def dbpedia_spotlight_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def custom_relation_linking_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def custom_entity_linking_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass