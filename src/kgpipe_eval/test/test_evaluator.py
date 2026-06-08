from __future__ import annotations

from kgpipe_eval.evaluator import Evaluator
from kgpipe_eval.metrics.statistics import CountMetric
from kgpipe_eval.metrics.duplicates import DuplicateMetric, DuplicateConfig
from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig
from kgpipe_eval.test.utils import get_test_kg, get_verified_entities_path
from kgpipe_eval.utils.kg_utils import KgManager


def test_evaluator_runs_metrics_with_and_without_config() -> None:
    kg = KgManager.load_kg(get_test_kg())
    try:
        dup_cfg = DuplicateConfig(
            entity_alignment_config=EntityAlignmentConfig(
                method="label_embedding",
                verified_entities_path=get_verified_entities_path(),
                verified_entities_delimiter=",",
                entity_sim_threshold=0.95,
            )
        )

        metrics = [CountMetric(), DuplicateMetric()]
        confs = {"DuplicateMetric": dup_cfg}

        results = Evaluator().run(kg=kg, metrics=metrics, confs=confs)
        assert len(results) == 2
        assert results[0].metric.__class__.__name__ == "CountMetric"
        assert results[1].metric.__class__.__name__ == "DuplicateMetric"
    finally:
        KgManager.unload_kg(kg)

