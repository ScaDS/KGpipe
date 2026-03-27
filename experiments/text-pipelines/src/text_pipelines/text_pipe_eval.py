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

TEST_NTRIPLES = """<http://example.org/bookstore/itemA> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/bookstore/Book> .
<http://example.org/bookstore/itemA> <http://www.w3.org/2000/01/rdf-schema#label> "itemA" .
<http://example.org/bookstore/itemA> <http://example.org/bookstore/bookTitle> "The Hobbit, or There and Back Again" .
<http://example.org/bookstore/itemA> <http://example.org/bookstore/bookAuthor> <http://example.org/bookstore/authorTolkien> .
<http://example.org/bookstore/itemA> <http://example.org/bookstore/isbn13> "9780261102217" .

<http://example.org/bookstore/itemB> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/bookstore/Book> .
<http://example.org/bookstore/itemB> <http://www.w3.org/2000/01/rdf-schema#label> "itemB" .
<http://example.org/bookstore/itemB> <http://example.org/bookstore/bookTitle> "Pride & Prejudice" .
<http://example.org/bookstore/itemB> <http://example.org/bookstore/bookAuthor> <http://example.org/bookstore/authorAusten> .
<http://example.org/bookstore/itemB> <http://example.org/bookstore/isbn13> "9780199535569" .

<http://example.org/bookstore/itemC> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/bookstore/Book> .
<http://example.org/bookstore/itemC> <http://www.w3.org/2000/01/rdf-schema#label> "itemC" .
<http://example.org/bookstore/itemC> <http://example.org/bookstore/bookTitle> "1984" .
<http://example.org/bookstore/itemC> <http://example.org/bookstore/bookAuthor> <http://example.org/bookstore/authorOrwell> .
<http://example.org/bookstore/itemC> <http://example.org/bookstore/isbn13> "9780452284234" .
"""

VERIFIED_ENTITIES = """dataset\tentity_id\tentity_label\tentity_type
dataset\thttp://example.org/bookstore/itemA\titemA\thttp://example.org/bookstore/Book
dataset\thttp://example.org/bookstore/itemB\titemB\thttp://example.org/bookstore/Book
dataset\thttp://example.org/bookstore/itemC\titemC\thttp://example.org/bookstore/Book
"""

def eval_example(tmp_path: Path):
    """Example: Evaluate a KG against a ground truth."""

    kg_path = tmp_path / "my_kg.nt"
    kg_path.write_text(TEST_NTRIPLES)

    verified_entities_path = tmp_path / "verified_entities.csv"
    verified_entities_path.write_text(VERIFIED_ENTITIES)

    empty_path = tmp_path / "empty.nt"
    empty_path.write_text("")

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
        REFERENCE_KG_PATH=kg_path,
        SEED_KG_PATH=empty_path,
        VERIFIED_SOURCE_ENTITIES=verified_entities_path
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
    tmp = tempfile.mkdtemp()
    st_res, sem_res, ref_res = eval_example(Path(tmp))

    assert st_res and sem_res and ref_res

    assert st_res.overall_score == 0.5
    assert sem_res.overall_score == 1/3
    assert ref_res.overall_score == 1.0
