import json

from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric
from kgpipe_eval.test.utils import get_test_kg, get_verified_entities_path, render_metric_result
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