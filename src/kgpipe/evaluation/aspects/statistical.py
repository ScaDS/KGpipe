"""
Statistical Evaluation Aspect

Evaluates statistical properties of knowledge graphs.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from pydantic import BaseModel
from rdflib import Graph, URIRef, Literal, RDF, RDFS
import time

from kgpipe.common.registry import Registry

from ...common.models import KG
from ..base import EvaluationAspect, AspectResult, AspectEvaluator, Metric, MetricResult, MetricConfig
from .func.namespace import count_namespace_usage

from kgpipe.common.systemgraph import kg_class, kg_function

ACCEPTED_FORMATS = ['ttl', 'rdf', 'jsonld', 'nt', 'json']

class StatisticalConfig(MetricConfig):
    """Config for statistical metrics."""
    entity_count: int = 1000
    relation_count: int = 50
    triple_count: int = 100000
    class_count: int = 3
    class_occurrence: int = 3
    relation_occurrence: int = 50
    property_occurrence: int = 50
    namespace_usage: int = 10000

@Registry.metric()
@kg_class("EvalMetric")
class EntityCountMetric(Metric):
    """Count the number of unique entities in the KG."""
    
    def __init__(self):
        super().__init__(
            name="entity_count",
            description="Number of unique entities in the knowledge graph",
            aspect=EvaluationAspect.STATISTICAL
        )
    
    # @kg_function
    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:
        """Compute entity count from KG file."""
        try:
            # Simple implementation - count unique subjects and objects
            entities = set()
            
            graph = kg.get_graph()
            for s, p, o in graph:
                entities.add(str(s))
                if isinstance(o, URIRef):
                    entities.add(str(o))
            
            count = len(entities)
            # Normalize: assume 1000+ entities is good, 0 is bad
            normalized_score = self.normalize_score(count, config.entity_count, 1000)
            
            return MetricResult(
                name=self.name,
                value=float(count),
                normalized_score=normalized_score,
                details={"unique_entities": count},
                aspect=self.aspect
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                normalized_score=0.0,
                details={"error": str(e)},
                aspect=self.aspect
            )

class RelationCountMetricDetails(BaseModel):
    unique_relations: int
    relations: int


@Registry.metric()
@kg_class("EvalMetric")
class RelationCountMetric(Metric):
    """Count the number of unique relations in the KG."""
    
    def __init__(self):
        super().__init__(
            name="relation_count",
            description="Number of unique relations in the knowledge graph",
            aspect=EvaluationAspect.STATISTICAL
        )
    
    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:
        """Compute relation count from KG file."""
        try:
            relations = defaultdict(int)
            
            graph = kg.get_graph()
            for s, p, o in graph:
                relations[str(p)] += 1
            
            count = len(relations)
            sum_relations = sum(relations.values())
            # Normalize: assume 50+ relations is good, 0 is bad
            normalized_score = self.normalize_score(count, 0, config.relation_count)
            
            return MetricResult(
                name=self.name,
                value=float(count),
                normalized_score=normalized_score,
                details={"unique_relations": count, "relations": relations},
                aspect=self.aspect
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                normalized_score=0.0,
                details={"error": str(e)},
                aspect=self.aspect
            )


@Registry.metric()
@kg_class("EvalMetric")
class TripleCountMetric(Metric):
    """Count the total number of triples in the KG."""
    
    def __init__(self):
        super().__init__(
            name="triple_count",
            description="Total number of triples in the knowledge graph",
            aspect=EvaluationAspect.STATISTICAL
        )
    
    @kg_function
    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:
        """Compute triple count from KG file."""
        try:
            count = 0
            
            if kg.format.value in ACCEPTED_FORMATS:
                with open(kg.path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and not line.strip().startswith('#'):
                            parts = line.split()
                            if len(parts) >= 3:
                                count += 1
            elif kg.format.value == 'json':
                with open(kg.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        # Count triples in nested structure
                        count = self._count_triples_in_dict(data)
            
            # Normalize: assume 10000+ triples is good, 0 is bad
            normalized_score = self.normalize_score(count, 0, config.triple_count)
            
            return MetricResult(
                name=self.name,
                value=float(count),
                normalized_score=normalized_score,
                details={"total_triples": count},
                aspect=self.aspect
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                normalized_score=0.0,
                details={"error": str(e)},
                aspect=self.aspect
            )
    
    def _count_triples_in_dict(self, data: Dict) -> int:
        """Recursively count triples in nested dictionary."""
        count = 0
        for key, value in data.items():
            if isinstance(value, dict):
                count += self._count_triples_in_dict(value)
            elif isinstance(value, list):
                count += len(value)
            else:
                count += 1
        return count

@Registry.metric()
@kg_class("EvalMetric")
class ClassCountMetric(Metric):
    """Count the number of unique classes in the KG."""
    
    def __init__(self):
        super().__init__(
            name="class_count",
            description="Number of unique classes in the knowledge graph",
            aspect=EvaluationAspect.STATISTICAL
        )

    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:
        """Compute class count from KG file."""
        try:
            graph = kg.get_graph()
            # count each class
            classes = defaultdict(int)
            for s, p, o in graph:
                if p == URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type") or str(p).endswith("c74e2b735dd8dc85ad0ee3510c33925f"):
                    classes[str(o)] += 1
            sum_classes = len(classes)
            normalized_score = self.normalize_score(sum_classes, 0, config.class_count)
            return MetricResult(
                name=self.name,
                value=float(sum_classes),
                normalized_score=normalized_score,
                details={"unique_classes": sum_classes, "classes": classes},
                aspect=self.aspect
            )
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                normalized_score=0.0,
                details={"error": str(e)},
                aspect=self.aspect
            )
            
@Registry.metric()
@kg_class("EvalMetric")
class ClassOccurrenceMetric(Metric):

    def __init__(self):
        super().__init__(
            name="class_occurrence",
            description="Occurrence of classes in the knowledge graph",
            aspect=EvaluationAspect.STATISTICAL
        )
    
    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:
        graph = kg.get_graph()
        classes = defaultdict(int)
        for s, p, o in graph.triples((None, RDF.type, None)):
            if isinstance(o, URIRef):
                classes[str(o)] += 1
        sum_classes = len(classes)  

        return MetricResult(
            name=self.name,
            value=float(sum_classes),
            normalized_score=0.0,
            details={"unique_classes": sum_classes, "classes": classes},
            aspect=self.aspect
        )

@Registry.metric()
@kg_class("EvalMetric")
class RelationOccurrenceMetric(Metric):
    def __init__(self):
        super().__init__(
            name="relation_occurrence",
            description="Occurrence of relations in the knowledge graph",
            aspect=EvaluationAspect.STATISTICAL
        )
    
    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:

        graph = kg.get_graph()
        relations = defaultdict(int)
        for s, p, o in graph:
            if isinstance(o, URIRef):
                relations[str(p)] += 1
        sum_relations = len(relations)

        return MetricResult(
            name=self.name,
            value=float(sum_relations),
            normalized_score=0.0,
            details={"unique_relations": sum_relations, "relations": relations},
            aspect=self.aspect
        )

@Registry.metric()
@kg_class("EvalMetric")
class PropertyOccurrenceMetric(Metric):
    def __init__(self):
        super().__init__(
            name="property_occurrence",
            description="Occurrence of properties in the knowledge graph",
            aspect=EvaluationAspect.STATISTICAL
        )
    
    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:
        graph = kg.get_graph()
        properties = defaultdict(int)
        for s, p, o in graph:
            if isinstance(o, Literal):
                properties[str(p)] += 1
        sum_properties = len(properties)

        return MetricResult(
            name=self.name,
            value=float(sum_properties),
            normalized_score=0.0,
            details={"unique_properties": sum_properties, "properties": properties},
            aspect=self.aspect
        )

@Registry.metric()
@kg_class("EvalMetric")
class NamespaceUsageMetric(Metric):

    def __init__(self):
        super().__init__(
            name="namespace_usage",
            description="Usage of namespaces in the knowledge graph",
            aspect=EvaluationAspect.STATISTICAL
        )
    
    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:

        graph = kg.get_graph()
        uris = list(set([str(s) for s in graph.subjects() if isinstance(s, URIRef)] + 
        [str(o) for o in graph.objects() if isinstance(o, URIRef)] + 
        [str(p) for p in graph.predicates() if isinstance(p, URIRef)]))
        namespace_usage = count_namespace_usage(uris)


        return MetricResult(
            name=self.name,
            value=float(sum(namespace_usage.values())),
            normalized_score=0.0, # TODO: normalize
            details={"namespace_usage": namespace_usage},
            aspect=self.aspect
        )

@kg_class("EvalMetric")
class LooseEntityCountMetric(Metric):
    def __init__(self):
        super().__init__(
            name="loose_entity_count",
            description="Number of entities in the KG not having any relations",
            aspect=EvaluationAspect.STATISTICAL
        )

    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:
        graph = kg.get_graph()
        entities = set()

        o_uris = [str(o) for _, p, o in graph if isinstance(o, URIRef) and p not in {RDFS.label, RDF.type}]
        s_uris = [str(s) for s in graph.subjects() if isinstance(s, URIRef)]
        all_entities = set(s_uris + o_uris)

        o_not_in_s = set(o_uris) - set(s_uris)

        return MetricResult(
            name=self.name,
            value=float(len(o_not_in_s)),
            normalized_score=1 - (len(o_not_in_s) / len(all_entities)),
            details={"o_not_in_s": o_not_in_s},
            aspect=self.aspect
        )

@kg_class("EvalMetric")
class ShallowEntityCountMetric(Metric):
    def __init__(self):
        super().__init__(
            name="shallow_entity_count",
            description=(
                "Number of entities in the KG having only label and/or type information "
                "(only s rdfs:label o and/or s rdf:type o)"
            ),
            aspect=EvaluationAspect.STATISTICAL
        )

    def compute(self, kg: KG, config: StatisticalConfig, **kwargs) -> MetricResult:
        graph = kg.get_graph()

        # Track all entity URIs
        all_entities = set()
        # Track entities with additional properties beyond rdfs:label and rdf:type
        entities_with_extra_info = set()

        for s, p, o in graph:
            if isinstance(s, URIRef):
                all_entities.add(str(s))
                # If predicate is not label or type, then it's "extra info"
                if p not in {RDFS.label, RDF.type}:
                    entities_with_extra_info.add(str(s))

        # Shallow = all_entities - entities_with_extra_info
        shallow_entities = all_entities - entities_with_extra_info

        return MetricResult(
            name=self.name,
            value=float(len(shallow_entities)),
            normalized_score=1 - (len(shallow_entities) / len(all_entities)),
            details={"entities": list(map(str, shallow_entities))},
            aspect=self.aspect
        )

class StatisticalEvaluator(AspectEvaluator):
    """Evaluator for statistical aspects of knowledge graphs."""
    
    def __init__(self):
        super().__init__(EvaluationAspect.STATISTICAL)
        self.metrics = [
            EntityCountMetric(),
            RelationCountMetric(),
            TripleCountMetric(),
            ClassCountMetric(),
            ClassOccurrenceMetric(),
            RelationOccurrenceMetric(),
            PropertyOccurrenceMetric(),
            NamespaceUsageMetric(),
            LooseEntityCountMetric(),
            ShallowEntityCountMetric()
        ]
    
    def evaluate(self, kg: KG, metrics: Optional[List[str]] = None, config: Optional[StatisticalConfig] = None, **kwargs) -> AspectResult:
        """Evaluate statistical properties of the KG."""
        results = []
        
        # Filter metrics if specified
        metrics_to_compute = self.metrics
        if metrics:
            metrics_to_compute = [m for m in self.metrics if m.name in metrics]
        
        # Compute each metric
        for metric in metrics_to_compute:
            try:
                if config is None:
                    config = StatisticalConfig(name="default")
                start_time = time.time()
                result = metric.compute(kg, config, **kwargs)
                end_time = time.time()
                result.duration = end_time - start_time
                results.append(result)
            except Exception as e:
                # Create error result
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
                "successful_metrics": len([r for r in results if "error" not in r.details])
            }
        )
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available statistical metrics."""
        return [metric.name for metric in self.metrics] 