from kgpipe_eval.metrics import CountMetric, DuplicateMetric
from typing import List
from kgpipe_eval.api import MetricConfig, MetricResult
from kgpipe_eval.metrics.statistics import CountMetric
from kgpipe_eval.metrics.duplicates import DuplicateConfig, DuplicateMetric
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric
from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig
from kgpipe_eval.utils.kg_utils import KgLike, KgManager
from kgpipe_eval.evaluator import Evaluator
from pydantic import BaseModel, ConfigDict

from kgpipe.datasets.multipart_multisource import Dataset, load_dataset
from kgpipe_eval.test.utils import render_metric_result
from pathlib import Path
import pytest
from kgpipe.common.model.pipeline import KgPipePlan, KgPipeReport
from kgpipe.common.model.kg import KG
from kgpipe.common.model.data import DataFormat
import json
# TODO 
# [ ] Dataset Reader (split,ref,source,metadata)
# [ ] Pipeline Results Reader (stage,kg,plan,report,tmp_file)


# TODO clearify
# substract seed from kg_1 and kg_1 from kg_2, or only seed from kg_1 and kg_2


EX_BENCH_DATA_PATH = Path("/home/marvin/phd/data/moviekg/datasets/film_10k")
EX_INC_PIPE_DATA_PATH = Path("/home/marvin/phd/data/moviekg/output/large/rdf_a")

# TODO is a wrapper interface for now, Dataset needs refactor later
# TODO can be abstracted and implemented to have direct method per type, so dict is not needed for access
class KgBenchData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset: Dataset
    
    @staticmethod
    def from_path(path: Path) -> 'KgBenchData':
        dataset = load_dataset(path)
        return KgBenchData(dataset=dataset)

    def get_verified_entities_path(self, i: int, source_type: str) -> Path:
        current_path = self.dataset.splits[f"split_{i}"].kg_reference.meta.entities.file
        current_new = current_path.with_name(f"{current_path.stem}_no_seed{current_path.suffix}")
        return current_new

class KgPipeData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    result_kg: KgLike # name=rdf_a_1
    plan: KgPipePlan
    report: KgPipeReport
    tmp_dir: Path

    @staticmethod
    def from_path(path: Path | str) -> 'KgPipeData':
        path = Path(path)
        plan = KgPipePlan.from_path(path / "exec-plan.json")
        report = KgPipeReport.from_path(path / "exec-report.json")
        tmp_dir = path / "tmp"
        return KgPipeData(
            result_kg=KG(name=path.name, id=path.name, path=path / "result.nt", format=DataFormat.RDF_NTRIPLES),
            plan=plan,
            report=report,
            tmp_dir=tmp_dir
        )

def build_config_dict(i: int, pipe_data: KgPipeData, bench_data: KgBenchData) -> dict[str, MetricConfig]:
    dup_cfg = DuplicateConfig(
        entity_alignment_config=EntityAlignmentConfig(
            method="label_embedding",
            verified_entities_path=bench_data.get_verified_entities_path(i=i, source_type="todo"),
            verified_entities_delimiter="\t",
            entity_sim_threshold=0.95,
        )
    )

    ent_cfg = EntityAlignmentConfig(
        method="label_embedding_and_type",
        verified_entities_path=bench_data.get_verified_entities_path(i=i, source_type="rdf"), # TODO type needs to be derived from pipe_data
        verified_entities_delimiter="\t",
        entity_sim_threshold=0.95
    )

    return {
        "DuplicateMetric": dup_cfg,
        "EntityAlignmentMetric": ent_cfg,
    }


def evaluate_stage(i: int, pipe_data: KgPipeData, bench_data: KgBenchData) -> List[MetricResult]:
    tg = KgManager.load_kg(pipe_data.result_kg)
    metrics = [
        CountMetric(), 
        EntityAlignmentMetric(),
        DuplicateMetric()
    ]
    config_dict = build_config_dict(i, pipe_data, bench_data)
    return Evaluator().run(tg, metrics, config_dict)

# EX_PIPE_DATA_PATH = Path("/home/marvin/phd/data/moviekg/output/small/rdf_a/stage_1")
# def test_evaluate_stage():
#     if not EX_PIPE_DATA_PATH.exists() or not EX_BENCH_DATA_PATH.exists():
#         pytest.skip("Local MovieKG eval data not available; test is an integration/WIP scaffold.")
#     pipe_data = KgPipeData.from_path(EX_PIPE_DATA_PATH)
#     bench_data = KgBenchData.from_path(EX_BENCH_DATA_PATH)
#     results = evaluate_stage(1, pipe_data, bench_data)

#     # render
#     print() # avoids pytest output being interleaved with print statements
#     for result in results:
#         print(render_metric_result(result, truncate=True, truncate_value=3))

def test_evaluate_inc_stage():
    if not EX_INC_PIPE_DATA_PATH.exists() or not EX_BENCH_DATA_PATH.exists():
        pytest.skip("Local MovieKG inc data not available; test is an integration/WIP scaffold.")

    for i in range(1, 4):
        pipe_data = KgPipeData.from_path(EX_INC_PIPE_DATA_PATH / f"stage_{i}")
        bench_data = KgBenchData.from_path(EX_BENCH_DATA_PATH)
        results = evaluate_stage(i, pipe_data, bench_data)

        # render
        print() # avoids pytest output being interleaved with print statements
        for result in results:
            print(render_metric_result(result, truncate=True, truncate_value=3))