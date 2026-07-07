from kgpipe_eval.evaluator import Evaluator
from kgpipe_eval.utils.kg_utils import KgLike, KgManager
from kgpipe_eval.utils.score_utils import aggregate_scores_from_json, aggregate_scores_from_results
from kgpipe_search.definitions import PipelineConfig

aggregation_config = {
  "subgroups": {
    "coverage": {
      "measurements": [
        {"metric": "EntityAlignmentMetric", "measurement": "recall"},
        {"metric": "TripleAlignmentMetric", "measurement": "recall"}
      ],
      "aggregation": "mean"
    },
    "correctness": {
      "measurements": [
        "EntityAlignmentMetric.precision",
        "TripleAlignmentMetric.precision"
      ],
      "aggregation": "mean"
    },
    # "consistency": {
    #   "measurements": [
    #     {"metric": "ConsistencyMetric", "measurement": "consistency_score"}
    #   ],
    #   "aggregation": "mean"
    # },
    # "cleanliness": {
    #   "measurements": [
    #     {"metric": "DuplicateMetric", "measurement": "duplicates_ratio", "transform": "invert"}
    #   ],
    #   "aggregation": "mean"
    # }
  },
  "final": {
    "aggregation": "weighted_mean",
    "weights": {
      "coverage": 0.5,
      "correctness": 0.5,
    #   "cleanliness": 0.2
    }
  }
}

def test_aggregate_results():
    result = aggregate_scores_from_json('data/eval_results.json', aggregation_config)
    print(f'Final score: {result.final_score:.6f}')
    for name, sg in result.subgroups.items():
        print(f'  {name}: {sg.score:.6f}')
        for m in sg.measurements:
            print(f'    {m.metric}.{m.measurement} = {m.value:.6f}')

def evaluate_pipeline(pipeline_config: PipelineConfig, result_kg: KgLike, reference_kg: KgLike):
    from kgpipe_eval.metrics.triple_alignment import TripleAlignmentMetric, TripleAlignmentConfig
    from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric, EntityAlignmentConfig

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

    result_graph = KgManager.load_kg(result_kg)
    try:
        results = Evaluator().run(result_graph, [TripleAlignmentMetric(), EntityAlignmentMetric()], {
            "TripleAlignmentMetric": triple_alignment_config,
            "EntityAlignmentMetric": entity_alignment_config
        })
    finally:
        KgManager.unload_kg(result_graph)

    return aggregate_scores_from_results(results, aggregation_config)


import random

def dummy_evaluate_pipeline(pipeline_config: PipelineConfig, result_kg: KgLike, reference_kg: KgLike):
    return random.uniform(0.5, 1.0) # 0.5 to 1.0

def _execute_pipeline(pipeline_config: PipelineConfig):
    pass

def execute_and_dummy_evaluate_pipeline(pipeline_config: PipelineConfig):
  result = _execute_pipeline(pipeline_config)
  return dummy_evaluate_pipeline(pipeline_config, None, None)