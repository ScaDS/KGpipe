from typing import List
from kgpipe_eval.api import MetricConfig, MetricResult
from kgpipe_eval.metrics.statistics import CountMetric
from kgpipe_eval.metrics.duplicates import DuplicateConfig, DuplicateMetric
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric
from kgpipe_eval.metrics.triple_alignment import TripleAlignmentConfig, TripleAlignmentMetric
from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig
from kgpipe_eval.utils.kg_utils import KgLike, KgManager
from kgpipe_eval.evaluator import Evaluator
from pydantic import BaseModel, ConfigDict

from kgpipe.datasets.multipart_multisource import Dataset, load_dataset
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
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

def build_config_dict(reference_kg) -> dict[str, MetricConfig]:
    dup_cfg = DuplicateConfig(
        entity_alignment_config=EntityAlignmentConfig(
            method="label_embedding",
            entity_sim_threshold=0.95,
            reference_kg=Path(reference_kg),
        )
    )

    tri_cfg = TripleAlignmentConfig(
        reference_kg=Path(reference_kg),
        entity_alignment_config=EntityAlignmentConfig(
            method="label_embedding",
            reference_kg=Path(reference_kg),
            entity_sim_threshold=0.95,
        ),
        value_sim_threshold=0.5,
        cache_literal_embeddings=True,
        cache_ref_literal_embeddings=True,
    )

    ent_cfg = EntityAlignmentConfig(
        method="label_embedding",
        entity_sim_threshold=0.95,
        ignored_entities=set(),
        reference_kg=Path(reference_kg),
    )

    return {
        "DuplicateMetric": dup_cfg,
        "EntityAlignmentMetric": ent_cfg,
        "TripleAlignmentMetric": tri_cfg,
    }


def evaluate_stage(kg_path, reference_kg) -> List[MetricResult]:
    tg = KgManager.load_kg(Path(kg_path))

    configs = build_config_dict(reference_kg)
    dup_config = configs["DuplicateMetric"]
    e_a_config = configs["EntityAlignmentMetric"]
    t_a_config = configs["TripleAlignmentMetric"]

    results = [
        CountMetric().compute(tg),
        EntityAlignmentMetric().compute(tg, e_a_config),
        DuplicateMetric().compute(tg, dup_config),
        TripleAlignmentMetric().compute(tg, t_a_config),
    ]

    return results


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


def main(kg_path: str, reference_kg: str, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = evaluate_stage(kg_path, reference_kg)

    eval_results = _metric_results_to_jsonable(results)

    result_file = output_path / "eval_results.json"
    with open(result_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Wrote results to {result_file}")

    assert isinstance(results, list)
    assert results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a knowledge graph against a reference KG."
    )
    parser.add_argument(
        "--kg-path",
        required=True,
        help="Path to the knowledge graph file to evaluate.",
    )
    parser.add_argument(
        "--reference-kg",
        required=True,
        help="Path to the reference knowledge graph file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where eval_results.json will be written.",
    )

    args = parser.parse_args()
    main(
        kg_path=args.kg_path,
        reference_kg=args.reference_kg,
        output_dir=args.output_dir,
    )