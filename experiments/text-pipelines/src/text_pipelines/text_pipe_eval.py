import os.path
import tempfile

from dotenv import load_dotenv

from kgpipe.common import DataFormat
from kgpipe.evaluation.aspects.statistical import (
    StatisticalEvaluator,
    StatisticalConfig,
    EntityCountMetric
)
from kgpipe.evaluation.aspects.semantic import (
    SemanticEvaluator,
    SemanticConfig,
    DisjointDomainMetric,
    IncorrectRelationDirectionMetric,
    IncorrectRelationRangeMetric,
    IncorrectRelationDomainMetric,
    IncorrectDatatypeMetric,
    IncorrectDatatypeFormatMetric,
)
from kgpipe.evaluation.aspects.reference import (
    ReferenceEvaluator,
    ReferenceConfig,
    SourceTypedEntityCoverageMetric,
    ReferenceTripleAlignmentMetric,
    ReferenceTripleAlignmentMetricSoftE,
    ReferenceTripleAlignmentMetricSoftEV,
)
from kgpipe.common.model.kg import KG
from typing import List
from pathlib import Path

def eval_example(kg_path: Path, reference_kg_path: Path, verified_source_entities_path: Path, seed_kg_path: Path):
    """Example: Evaluate a KG against a ground truth."""

    kg = KG(
        id="my_kg",
        name="My Knowledge Graph",
        path=kg_path,
        format=DataFormat.RDF_NTRIPLES
    )

    statistical_config = StatisticalConfig(name="default")
    semantic_config = SemanticConfig(name="default")
    reference_config = ReferenceConfig(
        name="default",
        REFERENCE_KG_PATH=reference_kg_path,
        SEED_KG_PATH=seed_kg_path,
        VERIFIED_SOURCE_ENTITIES=verified_source_entities_path
    )
    statistical_evaluator = StatisticalEvaluator()
    semantic_evaluator = SemanticEvaluator()
    reference_evaluator = ReferenceEvaluator()

    statistical_metrics: List[str] = [EntityCountMetric().name]
    semantic_metrics: List[str] = [DisjointDomainMetric().name, IncorrectRelationDirectionMetric().name, IncorrectRelationRangeMetric().name, IncorrectRelationDomainMetric().name, IncorrectDatatypeMetric().name, IncorrectDatatypeFormatMetric().name]
    reference_metrics: List[str] = [SourceTypedEntityCoverageMetric().name, ReferenceTripleAlignmentMetric().name, ReferenceTripleAlignmentMetricSoftE().name, ReferenceTripleAlignmentMetricSoftEV().name]

    statistical_results = statistical_evaluator.evaluate(
        kg, metrics=statistical_metrics, config=statistical_config)
    semantic_results = semantic_evaluator.evaluate(
        kg, metrics=semantic_metrics, config=semantic_config)
    reference_results = reference_evaluator.evaluate(
        kg, metrics=reference_metrics, config=reference_config)

    return statistical_results, semantic_results, reference_results

def test_eval():
    load_dotenv(Path(os.path.abspath(os.path.dirname(__file__))).parent.parent / ".env")
    tmp_path = Path(tempfile.mkdtemp())

    seed_kg_path = tmp_path / "empty.nt"
    seed_kg_path.write_text("")

    st_res, sem_res, ref_res = eval_example(
        Path(os.getenv("DATASET_PATH")) / "split_0" / "kg" / "seed" / "data.nt",
        Path(os.getenv("DATASET_PATH")) / "split_0" / "kg" / "reference" / "data.nt",
        Path(os.getenv("DATASET_PATH")) / "split_0" / "kg" / "seed" / "meta" / "verified_entities.csv",
        seed_kg_path
    )

    assert st_res and sem_res and ref_res

    assert st_res.overall_score == 0.5
    assert sem_res.overall_score == 1/3
    assert ref_res.overall_score > 0.9
