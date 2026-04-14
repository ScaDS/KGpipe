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
from dataclasses import asdict
from itertools import permutations
from typing import Set
from kgpipe_eval.utils.kg_utils import Term

try:
    from moviekg import config as moviekg_config
    from moviekg.pipelines.test_inc_msp import ssp, idfn
except Exception as e:
    # These are integration-style tests that depend on local env/config files.
    import traceback
    traceback.print_exc()
    pytest.skip(f"MovieKG config not available for eval integration test: {e}", allow_module_level=True)
# TODO 
# [ ] Dataset Reader (split,ref,source,metadata)
# [ ] Pipeline Results Reader (stage,kg,plan,report,tmp_file)


# TODO clearify
# substract seed from kg_1 and kg_1 from kg_2, or only seed from kg_1 and kg_2


EX_BENCH_DATA_PATH = Path("/home/marvin/phd/data/moviekg/datasets/film_10k") # TODO read from env
# EX_INC_PIPE_DATA_PATH = Path("/home/marvin/phd/data/moviekg/output/large/rdf_a")

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

    def get_ignored_entities(self, i: int, source_type: str) -> Set[Term]:
        seed_entities = self.dataset.splits[f"split_{0}"].kg_seed.meta.entities.read_csv()
        # source_seed_entities = self.dataset.splits[f"split_{i-1}"].sources[source_type].meta.entities.read_csv()
        return set([entity.entity_id for entity in seed_entities]) # + [entity.entity_id for entity in source_seed_entities])


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
            verified_entities_path=bench_data.get_verified_entities_path(i=i, source_type="rdf"), # TODO type needs to be derived from pipe_data
            verified_entities_delimiter="\t",
            entity_sim_threshold=0.95,
        )
    )

    ent_cfg = EntityAlignmentConfig(
        method="label_embedding_and_intersecting_type",
        verified_entities_path=bench_data.get_verified_entities_path(i=i, source_type="rdf"), # TODO type needs to be derived from pipe_data
        verified_entities_delimiter="\t",
        entity_sim_threshold=0.95,
        ignored_entities=bench_data.get_ignored_entities(i=i, source_type="rdf") # TODO type needs to be derived from pipe_data
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


def _stage_dirs(output_dir: Path) -> list[Path]:
    stage_dirs = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("stage_")]
    # stage_1, stage_2, ...
    stage_dirs.sort(key=lambda p: int(p.name.split("_", 1)[1]))
    return stage_dirs


def _metric_results_to_jsonable(results: list[MetricResult]) -> list[dict]:
    """
    Convert `MetricResult` dataclasses to JSON-serializable dicts.

    `MetricResult.metric` is an object instance, so we store its key/classname.
    """
    out: list[dict] = []
    for r in results:
        metric_key = getattr(r.metric, "key", None) or r.metric.__class__.__name__
        out.append(
            {
                "metric": metric_key,
                "summary": r.summary,
                "measurements": [asdict(m) for m in r.measurements],
            }
        )
    return out

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

# def test_evaluate_inc_stage():
#     if not EX_INC_PIPE_DATA_PATH.exists() or not EX_BENCH_DATA_PATH.exists():
#         pytest.skip("Local MovieKG inc data not available; test is an integration/WIP scaffold.")

#     for i in range(1, 4):
#         pipe_data = KgPipeData.from_path(EX_INC_PIPE_DATA_PATH / f"stage_{i}")
#         bench_data = KgBenchData.from_path(EX_BENCH_DATA_PATH)
#         results = evaluate_stage(i, pipe_data, bench_data)

#         # render
#         print() # avoids pytest output being interleaved with print statements
#         for result in results:
#             print(render_metric_result(result, truncate=True, truncate_value=3))

@pytest.mark.parametrize(
    "pipeline_name",
    list[str](moviekg_config.pipeline_types.keys()) + list[str](moviekg_config.llm_pipeline_types.keys()),
)
def test_evaluate_new(pipeline_name: str):
    """
    Boilerplate integration test that runs the new eval API for each pipeline
    output under `OUTPUT_ROOT/<pipeline_name>/stage_*`.
    """
    output_dir = moviekg_config.OUTPUT_ROOT / pipeline_name

    if not output_dir.exists():
        pytest.skip(f"Pipeline output directory {output_dir} not found")

    stage_dirs = _stage_dirs(output_dir)
    if not stage_dirs:
        pytest.skip(f"No stage directories found under {output_dir}")

    # Uses the dataset selected/configured via `moviekg.config` env vars.
    bench_data = KgBenchData.from_path(EX_BENCH_DATA_PATH)

    for stage_dir in stage_dirs:
        i = int(stage_dir.name.split("_", 1)[1])
        pipe_data = KgPipeData.from_path(stage_dir)
        results = evaluate_stage(i=i, pipe_data=pipe_data, bench_data=bench_data)

        eval_results = _metric_results_to_jsonable(results)
        with open(stage_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
            print(f"Wrote results to {stage_dir / 'eval_results.json'}")

        # Smoke checks: we got metric results back for this stage.
        assert isinstance(results, list)
        assert results


@pytest.mark.parametrize(
    "source_1, source_2, source_3",
    permutations(list[str](ssp.keys()), 3),
    ids=idfn,
)
def test_evaluate_new_multisource_pipeline(source_1: str, source_2: str, source_3: str):
    """
    Integration test for the *multi-source* incremental pipelines where the selected
    source changes per iteration/stage (e.g. `a_b_c/stage_1`, `a_b_c/stage_2`, ...).
    """
    pipeline_name = f"{source_1}_{source_2}_{source_3}"
    output_dir = moviekg_config.OUTPUT_ROOT / pipeline_name

    if not output_dir.exists():
        pytest.skip(f"Pipeline output directory {output_dir} not found")

    stage_dirs = _stage_dirs(output_dir)
    if not stage_dirs:
        pytest.skip(f"No stage directories found under {output_dir}")

    bench_data = KgBenchData.from_path(EX_BENCH_DATA_PATH)

    for stage_dir in stage_dirs:
        i = int(stage_dir.name.split("_", 1)[1])
        pipe_data = KgPipeData.from_path(stage_dir)
        results = evaluate_stage(i=i, pipe_data=pipe_data, bench_data=bench_data)

        eval_results = _metric_results_to_jsonable(results)
        with open(stage_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        assert isinstance(results, list)
        assert results