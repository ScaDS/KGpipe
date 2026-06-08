from pathlib import Path
import json
import pytest

from kgpipe_eval.utils.kg_utils import KgManager
from kgpipe_eval.metrics.triple_alignment import TripleAlignmentMetric, TripleAlignmentConfig
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric, EntityAlignmentConfig
from kgpipe_eval.api import MetricResult
from kgpipe_eval.test.utils import render_metric_result

rdf_base_dir = Path("data/tmp/rdf_pipelines/")
text_base_dir = Path("data/tmp/text_pipelines/")
result_dir = Path("data/output/reference_eval")

def get_rdf_final_kgs():
    """
    get all files matching rdf_result_saved_sample_config_idx_*.nt in dir
    """
    BASE_DIR = rdf_base_dir
    return [f for f in BASE_DIR.glob("*eval.nt")]

def get_text_final_kgs():
    """
    get all files matching text_result_saved_sample_config_idx_*.nt in dir
    """
    BASE_DIR = text_base_dir
    return [f for f in BASE_DIR.glob("*eval.nt")]

def test_get_rdf_final_kgs():
    """
    test the get_final_kgs function
    """
    final_kgs = get_rdf_final_kgs()
    for final_kg in final_kgs:
        print(final_kg)

def test_get_text_final_kgs():
    """
    test the get_final_kgs function
    """
    final_kgs = get_text_final_kgs()
    for final_kg in final_kgs:
        print(final_kg)

def _write_to_file(string: str, path: Path):
    with open(path, "w") as f:
        f.write(string)
        print(f"wrote to {path}")

def _metric_result_to_jsonable(metric_result: MetricResult) -> dict:
    metric = metric_result.metric
    metric_key = getattr(metric, "key", metric.__class__.__name__)
    return {
        "metric": metric_key,
        "summary": metric_result.summary,
        "measurements": [
            {"name": m.name, "value": m.value, "unit": m.unit}
            for m in metric_result.measurements
        ],
    }

def _write_json(obj: object, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
        print(f"wrote to {path}")

# seed_kg = KgManager.load_kg(Path("data/input_final/target_kg/graph.nt"))

def eval_pipeline(final_kg, reference_kg_path):

    print(f"evaluating {final_kg}")

    ref_kg_path = reference_kg_path
    gen_kg_path = final_kg

    entity_alignment_config = EntityAlignmentConfig(
            method="label_embedding",
            reference_kg=ref_kg_path,
            verified_entities_path=None,
            verified_entities_delimiter="\t",
            entity_sim_threshold=0.95
    )

    
    gen_kg = KgManager.load_kg(gen_kg_path)
    # test_kg = KgManager.substract_kg(gen_kg, seed_kg) # TODO add back labels and types
    test_kg = gen_kg

    metric_result : MetricResult = EntityAlignmentMetric().compute(test_kg, entity_alignment_config)
    result_string = render_metric_result(metric_result)
    _write_to_file(result_string, result_dir / (final_kg.name + ".entity_alignment.txt"))
    _write_json(_metric_result_to_jsonable(metric_result), result_dir / (final_kg.name + ".entity_alignment.json"))


    triple_alignment_config = TripleAlignmentConfig(
        reference_kg=ref_kg_path,
        entity_alignment_config=entity_alignment_config,
        value_sim_threshold=0.5,
        cache_literal_embeddings=True
    )

    metric_result : MetricResult = TripleAlignmentMetric().compute(test_kg, triple_alignment_config)
    result_string = render_metric_result(metric_result)
    _write_to_file(result_string, result_dir / (final_kg.name + ".triple_alignment.txt"))
    _write_json(_metric_result_to_jsonable(metric_result), result_dir / (final_kg.name + ".triple_alignment.json"))

@pytest.mark.parametrize("final_kg", get_rdf_final_kgs())
def test_eval_rdf_pipeline_runs(final_kg):
    """
    evaluate all runs of the rdf pipelines
    """
    eval_pipeline(final_kg, Path("data/input_final/reference_kg/data_no_seed.nt"))

@pytest.mark.parametrize("final_kg", get_text_final_kgs())
def test_eval_text_pipeline_runs(final_kg):
    """
    evaluate all runs of the text pipelines
    """
    #  data/input_final/txt_source/ref

    eval_pipeline(final_kg, Path("/data/datasets/params_experiments/latest/input_final/txt_source/tmp_reference/reference_kg_noseed.nt"))