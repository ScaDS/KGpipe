import json

from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric
from kgpipe_eval.metrics.triple_alignment import TripleAlignmentMetric, TripleAlignmentConfig
from kgpipe_eval.test.utils import get_test_kg, get_verified_entities_path, render_metric_result, get_reference_kg, get_generated_kg
from kgpipe_eval.utils.kg_utils import KgManager
from kgpipe_eval.api import MetricResult


def test_align_entities_by_label_embedding():
    config = EntityAlignmentConfig(
        method="label_embedding",
        reference_kg=None,
        verified_entities_path=get_verified_entities_path(),
        verified_entities_delimiter=",",
        entity_sim_threshold=0.95
    )
    tg = KgManager.load_kg(get_test_kg())
    metric_result : MetricResult = EntityAlignmentMetric().compute(tg, config)
    print(render_metric_result(metric_result))

def test_align_entities_by_label_embedding_and_type():
    config = EntityAlignmentConfig(
        method="label_embedding_and_type",
        reference_kg=None,
        verified_entities_path=get_verified_entities_path(),
        verified_entities_delimiter=",",
        entity_sim_threshold=0.95
    )
    tg = KgManager.load_kg(get_test_kg())
    metric_result : MetricResult = EntityAlignmentMetric().compute(tg, config)
    print(render_metric_result(metric_result))

def test_align_entities_by_label_embedding_and_type_ref_kg():
    config = EntityAlignmentConfig(
        method="label_embedding",
        reference_kg=get_reference_kg(),
        verified_entities_path=None,
        verified_entities_delimiter="\t",
        entity_sim_threshold=0.95
    )
    tg = KgManager.load_kg(get_test_kg())
    metric_result : MetricResult = EntityAlignmentMetric().compute(tg, config)
    print(render_metric_result(metric_result))

def test_align_triples_by_value_embedding():
    config = TripleAlignmentConfig(
        reference_kg=get_reference_kg(),
        entity_alignment_config=EntityAlignmentConfig(
            method="label_embedding",
            reference_kg=get_reference_kg(),
            verified_entities_path=None,
            verified_entities_delimiter="\t",
            entity_sim_threshold=0.95
        ),
        value_sim_threshold=0.5
    )
    tg = KgManager.load_kg(get_generated_kg())
    metric_result : MetricResult = TripleAlignmentMetric().compute(tg, config)
    print(render_metric_result(metric_result))

