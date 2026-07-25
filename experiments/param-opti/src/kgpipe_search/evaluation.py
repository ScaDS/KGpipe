from __future__ import annotations

from typing import Any, Mapping

from kgpipe_eval.evaluator import Evaluator
from kgpipe_eval.utils.kg_utils import KgLike, KgManager
from kgpipe_eval.utils.metric_utils import MeasurementKey
from kgpipe_eval.utils.score_utils import (
    AggregateScore,
    aggregate_scores,
    aggregate_scores_from_json,
    aggregate_scores_from_results,
)
from kgpipe_search.definitions import PipelineConfig
from kgpipe_search.ranking_conf import DEFAULT_AGGREGATION_CONFIG, get_aggregation_config
import os

# Backwards-compatible alias for the historical default aggregation.
aggregation_config = DEFAULT_AGGREGATION_CONFIG


def test_aggregate_results():
    result = aggregate_scores_from_json('data/eval_results.json', aggregation_config)
    print(f'Final score: {result.final_score:.6f}')
    for name, sg in result.subgroups.items():
        print(f'  {name}: {sg.score:.6f}')
        for m in sg.measurements:
            print(f'    {m.metric}.{m.measurement} = {m.value:.6f}')


def measurements_from_cached_evaluation(evaluation: Mapping[str, Any]) -> dict[MeasurementKey, float]:
    """Extract raw metric measurements from a cached AggregateScore JSON payload."""
    lookup: dict[MeasurementKey, float] = {}
    subgroups = evaluation.get("subgroups")
    if not isinstance(subgroups, Mapping):
        return lookup
    for subgroup in subgroups.values():
        if not isinstance(subgroup, Mapping):
            continue
        measurements = subgroup.get("measurements")
        if not isinstance(measurements, list):
            continue
        for item in measurements:
            if not isinstance(item, Mapping):
                continue
            metric = item.get("metric")
            measurement = item.get("measurement")
            value = item.get("value")
            if not isinstance(metric, str) or not isinstance(measurement, str):
                continue
            if not isinstance(value, (int, float)):
                continue
            lookup[MeasurementKey(metric=metric, measurement=measurement, unit="")] = float(value)
    return lookup


def aggregate_from_cached_evaluation(
    evaluation: Mapping[str, Any],
    config: Mapping[str, Any] | str | None = None,
) -> AggregateScore:
    """
    Re-aggregate a cached eval snapshot with ``config``.

    ``config`` may be an aggregation dict or a named config from ranking_conf
    (``default``, ``flat_hmean``). Defaults to the historical subgroup aggregation.
    Falls back to the stored ``final_score`` when measurements are missing.
    """
    if config is None:
        resolved = DEFAULT_AGGREGATION_CONFIG
    elif isinstance(config, str):
        resolved = get_aggregation_config(config)
    else:
        resolved = config

    lookup = measurements_from_cached_evaluation(evaluation)
    if not lookup:
        final_score = evaluation.get("final_score")
        if isinstance(final_score, (int, float)):
            return AggregateScore(final_score=float(final_score))
        raise ValueError("cached evaluation has neither measurements nor final_score")

    return aggregate_scores(lookup, resolved)


def score_from_cached_evaluation(
    evaluation: Mapping[str, Any],
    config: Mapping[str, Any] | str | None = None,
) -> float:
    """Convenience wrapper returning only the final score."""
    return float(aggregate_from_cached_evaluation(evaluation, config).final_score)


def evaluate_pipeline(
    pipeline_config: PipelineConfig,
    result_kg: KgLike,
    reference_kg: KgLike,
    aggregation: Mapping[str, Any] | str | None = None,
):
    from kgpipe_eval.metrics.statistics import CountMetric
    from kgpipe_eval.metrics.triple_alignment import TripleAlignmentMetric, TripleAlignmentConfig
    from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric, EntityAlignmentConfig
    from kgpipe_eval.metrics.consistency_violations import ConsistencyViolationsConfig,DisjointDomainMetric, DomainMetric, RangeMetric, DatatypeFormatMetric, DatatypeMetric, RelationDirectionMetric

    from kgpipe_eval.utils.kg_utils import KgManager

    if aggregation is None:
        resolved_config = DEFAULT_AGGREGATION_CONFIG
    elif isinstance(aggregation, str):
        resolved_config = get_aggregation_config(aggregation)
    else:
        resolved_config = aggregation

    source_seed_path: KgLike = os.getenv("SOURCE_SEED_PATH")
    source_seed_graph = KgManager.load_kg(source_seed_path)
    result_graph = KgManager.load_kg(result_kg)
    result_no_seed_graph = KgManager.substract_kg(result_graph, source_seed_graph)

    # Empty after seed subtract: alignment encode/dot and some consistency metrics break.
    if len(result_no_seed_graph.get_graph()) == 0:
        KgManager.unload_kg(result_graph)
        KgManager.unload_kg(result_no_seed_graph)
        return AggregateScore(final_score=0.0)

    consistency_violations_config = ConsistencyViolationsConfig(
        reference_kg=None,
        ontology_path=os.getenv("ONTOLOGY_PATH")
    )

    entity_alignment_config = EntityAlignmentConfig(
        method="label_embedding",
        reference_kg=reference_kg,
        verified_entities_path=None,
        verified_entities_delimiter="\t",
        entity_sim_threshold=0.95
    )

    triple_alignment_config = TripleAlignmentConfig(
        reference_kg=reference_kg,
        entity_alignment_config=entity_alignment_config,
        value_sim_threshold=0.5,
        cache_literal_embeddings=True
    )

    try:
        results = Evaluator().run(result_no_seed_graph, [TripleAlignmentMetric(), EntityAlignmentMetric(), CountMetric(), DisjointDomainMetric(), DomainMetric(), RangeMetric(), DatatypeFormatMetric(), DatatypeMetric(), RelationDirectionMetric()], {
            "TripleAlignmentMetric": triple_alignment_config,
            "EntityAlignmentMetric": entity_alignment_config,
            "DisjointDomainMetric": consistency_violations_config,
            "DomainMetric": consistency_violations_config,
            "RangeMetric": consistency_violations_config,
            "DatatypeFormatMetric": consistency_violations_config,
            "DatatypeMetric": consistency_violations_config,
            "RelationDirectionMetric": consistency_violations_config
        })
    finally:
        KgManager.unload_kg(result_graph)
        KgManager.unload_kg(result_no_seed_graph)

    return aggregate_scores_from_results(results, resolved_config)


import random

def dummy_evaluate_pipeline(pipeline_config: PipelineConfig, result_kg: KgLike, reference_kg: KgLike):
    return random.uniform(0.5, 1.0) # 0.5 to 1.0

def _execute_pipeline(pipeline_config: PipelineConfig):
    pass

def execute_and_dummy_evaluate_pipeline(pipeline_config: PipelineConfig):
  result = _execute_pipeline(pipeline_config)
  return dummy_evaluate_pipeline(pipeline_config, None, None)
