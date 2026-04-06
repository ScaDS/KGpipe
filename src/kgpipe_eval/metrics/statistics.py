from kgpipe.common import KG
from kgpipe_eval.utils.kg_utils import TripleGraph, KgManager
from kgpipe_eval.api import Metric, MetricResult, Measurement
from functools import lru_cache

from pydantic import BaseModel
from typing import Mapping
from collections import defaultdict

from rdflib import RDF, RDFS
from rdflib.term import URIRef, Literal

class CountMeasures(BaseModel):
    entity_count: int
    triple_count: int
    property_count: int
    class_count: int
    property_occurrence: Mapping[str, int]
    class_occurrence: Mapping[str, int]

@lru_cache(maxsize=1)
def count_measures(kg: TripleGraph) -> CountMeasures:
    
    entity_count = 0 # TODO requires distinct entities
    triple_count = 0

    class_occurrence = defaultdict(int)
    property_occurrence = defaultdict(int)
    
    for s, p, o in kg.triples((None, None, None)):
        triple_count += 1
        if p == RDF.type:
            class_occurrence[str(o)] += 1
        property_occurrence[str(p)] += 1

    return CountMeasures(
        entity_count=entity_count,
        property_count=len(property_occurrence.keys()),
        triple_count=triple_count,
        class_count=len(class_occurrence.keys()),
        class_occurrence=class_occurrence,
        property_occurrence=property_occurrence,
    )

class CountMetric(Metric):
    def compute(self, kg: TripleGraph) -> MetricResult:
        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="entity_count", value=count_measures(kg).entity_count, unit="number"),
                Measurement(name="triple_count", value=count_measures(kg).triple_count, unit="number"),
                Measurement(name="property_count", value=count_measures(kg).property_count, unit="number"),
                Measurement(name="class_count", value=count_measures(kg).class_count, unit="number"),
                Measurement(name="property_occurrence", value=count_measures(kg).property_occurrence, unit="number"),
                Measurement(name="class_occurrence", value=count_measures(kg).class_occurrence, unit="number"),
            ],
            summary=f"Measures of entities, triples, properties, classes, property occurrences, and class occurrences"
        )

