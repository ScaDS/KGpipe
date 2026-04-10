from kgpipe_eval.metrics.duplicates import DuplicateConfig
from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig
from kgpipe_eval.test.utils import get_verified_entities_path
from kgpipe_eval.api import MetricResult
from kgpipe_eval.metrics.duplicates import DuplicateMetric
from kgpipe_eval.test.utils import get_test_kg, render_metric_result
from kgpipe_eval.utils.kg_utils import KgManager

def test_duplicates_eval():
    config = DuplicateConfig(
        entity_alignment_config=EntityAlignmentConfig(
            method="label_embedding",
            reference_kg=None,
            verified_entities_path=get_verified_entities_path(),
            verified_entities_delimiter=",",
            entity_sim_threshold=0.95
        )
    )
    metric_result : MetricResult = DuplicateMetric().compute(KgManager.load_kg(get_test_kg()), config)
    print(render_metric_result(metric_result))
