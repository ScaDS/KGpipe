"""
Reference-based Evaluation Aspect

Evaluates knowledge graphs by comparing them against reference/gold standard KGs.
"""

from dataclasses import dataclass
from ntpath import isdir
import traceback
from typing import Dict, List, Optional, Any, Set
import json
from pathlib import Path
from rdflib import URIRef, Graph, RDF

from ...common.models import KG, Data, DataFormat
from ..base import EvaluationAspect, AspectResult, AspectEvaluator, Metric, MetricResult, MetricConfig
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document, ER_Match
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document
from kgpipe.evaluation.cluster import MatchCluster, is_match, load_matches
from enum import Enum
import time
from kgpipe.util.embeddings import global_encode, global_encode_as_dict

from kgpipe.evaluation.aspects.func.soft_metrics import (
    reference_alignment,
    reference_alignment_soft_entities,
    reference_alignment_soft_entities_values,
    graph_fact_alginment,
    graph_fact_alginment_soft_entities,
    graph_fact_alginment_soft_entities_values
)


from kgpipe.common.registry import Registry

from pyodibel.datasets.mp_mf.multipart_multisource import SourceMeta, Dataset, read_matches_csv

ONTOLOGY_REFERENCE_KEY = "ontology"

@dataclass # TODO in future, trust_level and associated_data can be important
class Reference:
    data: Data
    trust_level: float
    associated_data: Dict[str, Data]

class ReferenceConfig(MetricConfig):
    GT_MATCHES: Optional[Path] = None
    GT_MATCHES_TARGET_DATASET: Optional[str] = None
    ENTITY_MATCH_THRESHOLD: float = 0.5
    RELATION_MATCH_THRESHOLD: float = 0.5
    VERIFIED_SOURCE_ENTITIES: Optional[Path] = None
    REFERENCE_KG_PATH: Optional[Path] = None
    EXPECTED_TEXT_LINKS: Optional[Path] = None
    TE_LINK_THRESHOLD: float = 0.4
    SEED_KG_PATH: Optional[Path] = None
    source_meta: Optional[SourceMeta] = None
    dataset: Optional[Dataset] = None
    JSON_EXPECTED_DIR: Optional[str] = None
    JSON_EXPECTED_RELATION_FILE: Optional[str] = None

#==============================================
# Reference Metrics
#==============================================

@Registry.metric()
class ER_EntityMatchMetric(Metric):

    def __init__(self):
        super().__init__(
            name="ER_EntityMatchMetric",
            description="Evaluates the entity matching task",
            aspect=EvaluationAspect.REFERENCE
        )

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:
        
        # TODO change to verfied entities level
        # dataset = config.dataset
        # if dataset is None:
        #     raise ValueError("Dataset is not set")
        # gt_match_path = dataset.root / "split_match_entities.csv"

        gt_match_path = config.GT_MATCHES
        gt_match_target_dataset = config.GT_MATCHES_TARGET_DATASET
        if gt_match_path is None or gt_match_target_dataset is None:
            raise ValueError("GT_MATCHES or GT_MATCHES_TARGET_DATASET is not set")
        
        from kgpipe.evaluation.aspects.func.er_task_eval import evaluate_entity_matching

        value, normalized_score, details = evaluate_entity_matching(kg, gt_match_path, gt_match_target_dataset, config.ENTITY_MATCH_THRESHOLD)
        
        return MetricResult(
            name=self.name,
            value=value,
            normalized_score=normalized_score,
            details=details,
            aspect=self.aspect
        )


@Registry.metric()
class ER_RelationMatchMetric(Metric):
    def __init__(self):
        super().__init__(
            name="ER_RelationMatchMetric",
            description="Evaluates the relation matching task",
            aspect=EvaluationAspect.REFERENCE
        )
    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:
        from kgpipe.evaluation.aspects.func.er_task_eval import evaluate_relation_matching

        print(f"[CONFIG] Relation matching threshold: {config.RELATION_MATCH_THRESHOLD}")

        # TODO change to verfied entities level
        dataset = config.dataset
        if dataset is None:
            raise ValueError("Dataset is not set")
        gt_match_path = dataset.root / "split_match_entities.csv"
        
        value, normalized_score, details = evaluate_relation_matching(kg, gt_match_path, config.RELATION_MATCH_THRESHOLD)

        return MetricResult(
            name=self.name,
            value=value,
            normalized_score=normalized_score,
            details=details,
            aspect=self.aspect
        )

@Registry.metric()
class TE_ExpectedEntityLinkMetric(Metric):
    def __init__(self):
        super().__init__(
            name="TE_ExpectedEntityLinkMetric",
            description="Evaluates the expected entity links found in the text source",
            aspect=EvaluationAspect.REFERENCE
        )
    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:

        from kgpipe.evaluation.aspects.func.te_task_eval import evaluate_expected_entity_links

        meta = config.source_meta
        if meta is None or meta.links is None:
            raise ValueError("SourceMeta is not set")
        expected_text_links = meta.links.file

        link_counts = evaluate_expected_entity_links(expected_text_links, kg, config.TE_LINK_THRESHOLD)

        value = link_counts.true_link_cnt
        normalized_score = value / (link_counts.false_missing_link_cnt + link_counts.true_link_cnt) if link_counts.false_missing_link_cnt + link_counts.true_link_cnt > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=value,
            normalized_score=normalized_score,
            details=link_counts.__dict__,
            aspect=self.aspect
        )

@Registry.metric()
class TE_ExpectedRelationLinkMetric(Metric):
    def __init__(self):
        super().__init__(
            name="TE_ExpectedRelationLinkMetric",
            description="Evaluates the expected relation links found in the text source",
            aspect=EvaluationAspect.REFERENCE
        )
    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:

        from kgpipe.evaluation.aspects.func.te_task_eval import evaluate_expected_relation_links

        expected_text_links = config.EXPECTED_TEXT_LINKS
        if expected_text_links is None:
            raise ValueError("EXPECTED_TEXT_LINKS is not set")

        link_counts = evaluate_expected_relation_links(expected_text_links, kg, config.TE_LINK_THRESHOLD)

        value = link_counts.true_link_cnt
        normalized_score = value / (link_counts.true_link_cnt + link_counts.false_link_cnt) if link_counts.true_link_cnt + link_counts.false_link_cnt > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=value,
            normalized_score=normalized_score,
            details=link_counts.__dict__,
            aspect=self.aspect
        )

#==============================================
# JSON Specific Metrics
#==============================================

from kgpipe.evaluation.metrics.json_specific import get_json_mapping_prov, get_er_task_json_doc, evaluate_json_a_matching, evaluate_json_b_linking

@Registry.metric()  
class JsonEntityMatchingMetric(Metric):
    """
    Evaluation of entity matching through JSON-to-KG mapping.
    """
    def __init__(self):
        super().__init__(
            name="JsonEntityMatchingMetric",
            description="Evaluates the entity matching through JSON-to-KG mapping",
            aspect=EvaluationAspect.SPECIFIC
        )

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:
        """
        Compute the metric value for the given KG.
        """
        JSON_EXPECTED_DIR = config.JSON_EXPECTED_DIR
        JSON_ACTUAL_FILE = get_json_mapping_prov(kg, ".entity.prov")[0]
        MATCH_JSON_FILE = get_er_task_json_doc(kg)[0]
        MATCH_THRESHOLD = config.ENTITY_MATCH_THRESHOLD
        bin_class = evaluate_json_a_matching(JSON_EXPECTED_DIR, JSON_ACTUAL_FILE, MATCH_JSON_FILE, MATCH_THRESHOLD)

        return MetricResult(
            name=self.name,
            value=bin_class.f1_score(),
            normalized_score=bin_class.f1_score(),
            details=bin_class.__dict__(), # type: ignore
            aspect=self.aspect
        )

@Registry.metric()
class JsonRelationMatchingMetric(Metric):
    """
    Evaluation of relation matching through JSON-to-KG mapping.
    """
    def __init__(self):
        super().__init__(
            name="JsonRelationMatchingMetric",
            description="Evaluates the relation matching through JSON-to-KG mapping",
            aspect=EvaluationAspect.SPECIFIC
        )

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:
        """
        Compute the metric value for the given KG.
        """

        from kgpipe.evaluation.metrics.json_specific import evaluate_json_a_relation_matching
        # TODO later
        JSON_EXPECTED_RELATION_FILE = config.JSON_EXPECTED_RELATION_FILE
        JSON_ACTUAL_FILE = get_json_mapping_prov(kg, ".relation.prov")[0]
        MATCH_JSON_FILE = get_er_task_json_doc(kg)[0]
        MATCH_THRESHOLD = config.RELATION_MATCH_THRESHOLD
        bin_class = evaluate_json_a_relation_matching(JSON_EXPECTED_RELATION_FILE, JSON_ACTUAL_FILE, MATCH_JSON_FILE, MATCH_THRESHOLD)

        return MetricResult(
            name=self.name,
            value=bin_class.f1_score(),
            normalized_score=bin_class.f1_score(),
            details=bin_class.__dict__(), # type: ignore
            aspect=self.aspect
        )

@Registry.metric()
class JsonEntityLinkingMetric(Metric):
    """
    Evaluation of entity linking through JSON-to-KG mapping.
    """
    def __init__(self):
        super().__init__(
            name="JsonEntityLinkingMetric",
            description="Evaluates the entity linking through JSON-to-KG mapping",
            aspect=EvaluationAspect.SPECIFIC
        )
    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:
        """
        Compute the metric value for the given KG.
        """
        JSON_EXPECTED_DIR = config.JSON_EXPECTED_DIR
        JSON_ACTUAL_FILE = get_json_mapping_prov(kg, ".entity.prov")[0]
        bin_class = evaluate_json_b_linking(JSON_EXPECTED_DIR, JSON_ACTUAL_FILE)

        return MetricResult(
            name=self.name,
            value=bin_class.f1_score(),
            normalized_score=bin_class.f1_score(),
            details=bin_class.__dict__(), # type: ignore
            aspect=self.aspect
        )


# =============================================================================
# Source Integration (depends on source metadata, requires traceability)
# =============================================================================

@Registry.metric()
class SourceEntityCoverageMetric(Metric):
    def __init__(self):
        super().__init__(
            name="SourceEntityCoverageMetric",
            description="Evaluates the coverage of the source entities in the KG",
            aspect=EvaluationAspect.REFERENCE
        )

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:
        from kgpipe.evaluation.aspects.func.integration_eval import evaluate_source_entity_coverage

        # meta = config.source_meta
        # if meta is None or meta.entities is None:
        #     raise ValueError("SourceMeta is not set")
        verified_source_entities_path = config.VERIFIED_SOURCE_ENTITIES
        if verified_source_entities_path is None:
            raise ValueError("VERIFIED_SOURCE_ENTITIES is not set")
        
        result = evaluate_source_entity_coverage(kg, verified_source_entities_path)

        score = result.overlapping_entities_count / result.expected_entities_count

        return MetricResult(
            name=self.name,
            value=score if score < 1.0 else 1.0,
            normalized_score=score if score < 1.0 else 1.0,
            details=result.__dict__,
            aspect=self.aspect
        )

@Registry.metric()
class SourceEntityCoverageMetricSoft(Metric):
    def __init__(self):
        super().__init__(
            name="SourceEntityCoverageMetricSoft",
            description="Evaluates the coverage of the source entities in the KG, applying soft matching",
            aspect=EvaluationAspect.REFERENCE
        )

    # Match and map test KG entities to reference KG first
    # Then, evaluate the overlap between test and reference entities

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:

        from kgpipe.evaluation.aspects.func.integration_eval import evaluate_source_entity_coverage_fuzzy

        # meta = config.source_meta
        # if meta is None or meta.entities is None:
        #     raise ValueError("SourceMeta is not set")
        verified_source_entities_path = config.VERIFIED_SOURCE_ENTITIES
        if verified_source_entities_path is None:
            raise ValueError("VERIFIED_SOURCE_ENTITIES is not set")
        
        result = evaluate_source_entity_coverage_fuzzy(kg, verified_source_entities_path)

        # print("value", result.overlapping_entities_count / result.expected_entities_count)
        # print("normalized_score", result.overlapping_entities_count / result.expected_entities_count)
        # print("details", result.__dict__)

        score = result.overlapping_entities_count / result.expected_entities_count
        if score > 1.0:
            score = 1.0

        return MetricResult(
            name=self.name,
            value=score,
            normalized_score=score,
            details=result.__dict__,
            aspect=self.aspect
        )

@Registry.metric()
class SourceEntityPrecisionMetric(Metric):
    def __init__(self):
        super().__init__(
            name="SourceEntityPrecisionMetric",
            description="Evaluates the precision of the source entities in the KG, applying soft matching",
            aspect=EvaluationAspect.REFERENCE
        )

    # Match and map test KG entities to reference KG first
    # Then, evaluate the overlap between test and reference entities

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:

        from kgpipe.evaluation.aspects.func.integration_eval import evaluate_source_entity_precision_fuzzy, EntityCoverageResult

        # meta = config.source_meta
        # if meta is None or meta.entities is None:
        #     raise ValueError("SourceMeta is not set")
        verified_source_entities_path = config.VERIFIED_SOURCE_ENTITIES
        if verified_source_entities_path is None:
            raise ValueError("VERIFIED_SOURCE_ENTITIES is not set")
        
        result: EntityCoverageResult = evaluate_source_entity_precision_fuzzy(kg, verified_source_entities_path)

        # print("value", result.overlapping_entities_count / result.expected_entities_count)
        # print("normalized_score", result.overlapping_entities_count / result.expected_entities_count)
        # print("details", result.__dict__)

        duplicates = result.possible_duplicates_count
        found = result.found_entities_count

        score = (found - duplicates) / found if found > 0 else 0.0
        if score > 1.0:
            score = 1.0

        return MetricResult(
            name=self.name,
            value=score,
            normalized_score=score,
            details=result.__dict__,
            aspect=self.aspect
        )

# =============================================================================
# Reference KG Alignment
# =============================================================================

@Registry.metric()
class ReferenceTripleAlignmentMetric(Metric):
    def __init__(self):
        super().__init__(
            name="ReferenceTripleAlignmentMetric",
            description="Evaluates the alignment of the reference triples in the KG",
            aspect=EvaluationAspect.REFERENCE
        )

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:
        from kgpipe.evaluation.aspects.func.integration_eval import evaluate_reference_triple_alignment

        reference_kg_path = config.REFERENCE_KG_PATH
        if reference_kg_path is None:
            raise ValueError("REFERENCE_KG_PATH is not set")

        seed_kg_path = config.SEED_KG_PATH
        if seed_kg_path is None:
            raise ValueError("SEED_KG_PATH is not set")

        seed_graph = Graph()
        seed_graph.parse(config.SEED_KG_PATH.as_posix())

        seed_triples = set([str(s)+str(p)+str(o) for s, p, o in seed_graph])

        reference_graph_raw = Graph()
        reference_graph_raw.parse(reference_kg_path.as_posix())

        reference_graph = Graph()
        for s, p, o in reference_graph_raw:
            if str(s)+str(p)+str(o) in seed_triples:
                continue
            reference_graph.add((s, p, o))

        actual_graph_raw = kg.get_graph()
        actual_graph = Graph()
        for s, p, o in actual_graph_raw:
            if str(s)+str(p)+str(o) in seed_triples:
                continue
            actual_graph.add((s, p, o))

        print("graph_fact_alginment strict")

        result = graph_fact_alginment(actual_graph, reference_graph)

        return MetricResult(
            name=self.name,
            value=result.precision(),
            normalized_score=result.precision(),
            details=result.__dict__(),
            aspect=self.aspect
        )

@Registry.metric()
class ReferenceTripleAlignmentMetricSoftE(Metric):
    def __init__(self):
        super().__init__(
            name="ReferenceTripleAlignmentMetricSoftE",
            description="Evaluates the alignment of the reference triples in the KG",
            aspect=EvaluationAspect.REFERENCE
        )

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:

        from kgpipe.evaluation.aspects.func.ref_fuzzy import match_entities, map_entities

        tmp_stagging_kg_path = Path("/tmp/stagging_kg.nt")

        tmp_stagging_kg = KG(
            id="stagging_kg",
            name="stagging_kg",
            path=tmp_stagging_kg_path,
            format=DataFormat.RDF_NTRIPLES
        )

        reference_kg_path = config.REFERENCE_KG_PATH
        if reference_kg_path is None:
            raise ValueError("REFERENCE_KG_PATH is not set")

        seed_kg_path = config.SEED_KG_PATH
        if seed_kg_path is None:
            raise ValueError("SEED_KG_PATH is not set")

        seed_graph = Graph()
        seed_graph.parse(config.SEED_KG_PATH.as_posix())

        seed_triples = set([str(s)+str(p)+str(o) for s, p, o in seed_graph])

        reference_graph_raw = Graph()
        reference_graph_raw.parse(reference_kg_path.as_posix())

        reference_graph = Graph()
        for s, p, o in reference_graph_raw:
            if str(s)+str(p)+str(o) in seed_triples:
                continue
            reference_graph.add((s, p, o))

        actual_graph_raw = kg.get_graph()
        actual_graph = Graph()
        for s, p, o in actual_graph_raw:
            if str(s)+str(p)+str(o) in seed_triples:
                continue
            actual_graph.add((s, p, o))

        print("graph_fact_alginment_soft_entities")

        result = graph_fact_alginment_soft_entities(actual_graph, reference_graph)

        # matched_entities = match_entities(kg, reference_kg_path)
        # print("matched_entities", len([e for e in matched_entities.values() if e.decision == "accept"]))
        # mapped_graph = map_entities(matched_entities, kg.get_graph())
        # mapped_graph.serialize(destination=tmp_stagging_kg_path, format="nt")

        # result = ReferenceTripleAlignmentMetric().compute(tmp_stagging_kg, config)

        return MetricResult(
            name=self.name,
            value=result.precision(),
            normalized_score=result.f1_score(),
            details=result.__dict__(),
            aspect=self.aspect
        )

@Registry.metric()
class ReferenceTripleAlignmentMetricSoftEV(Metric):
    def __init__(self):
        super().__init__(
            name="ReferenceTripleAlignmentMetricSoftEV",
            description="Evaluates the alignment of the reference triples in the KG",
            aspect=EvaluationAspect.REFERENCE
        )

    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:

        from kgpipe.evaluation.aspects.func.integration_eval import evaluate_reference_triple_alignment_fuzzy
        from kgpipe.evaluation.aspects.func.ref_fuzzy import match_entities, map_entities

        reference_kg_path = config.REFERENCE_KG_PATH
        if reference_kg_path is None:
            raise ValueError("REFERENCE_KG_PATH is not set")

        seed_kg_path = config.SEED_KG_PATH
        if seed_kg_path is None:
            raise ValueError("SEED_KG_PATH is not set")

        seed_graph = Graph()
        seed_graph.parse(config.SEED_KG_PATH.as_posix())

        seed_triples = set([str(s)+str(p)+str(o) for s, p, o in seed_graph])

        reference_graph_raw = Graph()
        reference_graph_raw.parse(reference_kg_path.as_posix())

        reference_graph = Graph()
        for s, p, o in reference_graph_raw:
            if str(s)+str(p)+str(o) in seed_triples:
                continue
            if str(p) == RDF.type:
                continue
            reference_graph.add((s, p, o))

        actual_graph_raw = kg.get_graph()
        actual_graph = Graph()
        for s, p, o in actual_graph_raw:
            if str(s)+str(p)+str(o) in seed_triples:
                continue
            if str(p) == RDF.type:
                continue
            actual_graph.add((s, p, o))


        print("graph_fact_alginment_soft_entities_values")

        result = graph_fact_alginment_soft_entities_values(actual_graph , reference_graph)


        # matched_entities = match_entities(kg, reference_kg_path)
        # mapped_graph = map_entities(matched_entities, kg.get_graph())

        # reference_graph = Graph()
        # reference_graph.parse(data=reference_kg_path.as_posix())

        # result = evaluate_reference_triple_alignment_fuzzy(mapped_graph, reference_graph)

        return MetricResult(
            name=self.name,
            value=result.precision(),
            normalized_score=result.f1_score(),
            details=result.__dict__(),
            aspect=self.aspect
        )

@Registry.metric()
class ReferenceClassCoverageMetric(Metric):
    def __init__(self):
        super().__init__(
            name="ReferenceClassCoverageMetric",
            description="Evaluates the coverage of the reference classes in the KG",
            aspect=EvaluationAspect.REFERENCE
        )
    def compute(self, kg: KG, config: ReferenceConfig, **kwargs) -> MetricResult:
        # from kgpipe.evaluation.aspects.func.integration_eval import evaluate_reference_class_coverage
        # reference_kg_path = config.REFERENCE_KG_PATH
        # if reference_kg_path is None:
        #     raise ValueError("REFERENCE_KG_PATH is not set")
        # reference_kg = KG(
        #     id="reference_kg",
        #     name="reference_kg",
        #     path=reference_kg_path,
        #     format=DataFormat.RDF_TTL
        # )
        # result = evaluate_reference_class_coverage(kg, reference_kg)
        return MetricResult(
            name=self.name,
            value=0.0,
            normalized_score=0.0,
            details={"error": "Not implemented"},
            aspect=self.aspect
        )

class ReferenceEvaluator(AspectEvaluator):
    """Evaluator for reference-based aspects of knowledge graphs."""
    
    def __init__(self):
        super().__init__(EvaluationAspect.REFERENCE)
        self.metrics = [
            ER_EntityMatchMetric(),
            ER_RelationMatchMetric(),
            TE_ExpectedEntityLinkMetric(),
            TE_ExpectedRelationLinkMetric(),
            JsonEntityMatchingMetric(),
            JsonRelationMatchingMetric(),
            JsonEntityLinkingMetric(),
            SourceEntityCoverageMetric(),
            SourceEntityCoverageMetricSoft(),
            SourceEntityPrecisionMetric(),
            ReferenceTripleAlignmentMetric(),
            ReferenceTripleAlignmentMetricSoftE(),
            ReferenceTripleAlignmentMetricSoftEV(),
            # ReferenceClassCoverageMetric()
        ]
    
    def evaluate(self, kg: KG, config: ReferenceConfig, metrics: Optional[List[str]] = None, **kwargs) -> AspectResult:
        """Evaluate reference-based properties of the KG."""
        # if references is {}:
        #     # Return empty result if no reference KG provided
        #     return AspectResult(
        #         aspect=self.aspect,
        #         metrics=[],
        #         overall_score=0.0,
        #         details={"error": "No reference KG provided"}
        #     )

        if config is None:
            raise ValueError("ReferenceConfig is not set")
        
        results = []
        
        # Filter metrics if specified
        metrics_to_compute = self.metrics
        if metrics:
            metrics_to_compute = [m for m in self.metrics if m.name in metrics]
        
        # Compute each metric
        for metric in metrics_to_compute:
            try:
                start_time = time.time()
                result = metric.compute(kg, config, **kwargs)
                end_time = time.time()
                result.duration = end_time - start_time
                results.append(result)
            except Exception as e:
                print(f"[Error] computing metric {metric.name}: {e}")
                # Create error result
                # print stacktrace
                print(traceback.format_exc())
                error_result = MetricResult(
                    name=metric.name,
                    value=0.0,
                    normalized_score=0.0,
                    details={"error": str(e)},
                    aspect=self.aspect
                )
                results.append(error_result)
        
        # Calculate overall score as average of normalized scores
        if results:
            overall_score = sum(r.normalized_score for r in results) / len(results)
        else:
            overall_score = 0.0
        
        return AspectResult(
            aspect=self.aspect,
            metrics=results,
            overall_score=overall_score,
            details={
                "total_metrics": len(results),
                "successful_metrics": len([r for r in results if "error" not in r.details]),
                "references": config.__dict__
            }
        )
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available reference-based metrics."""
        return [metric.name for metric in self.metrics] 