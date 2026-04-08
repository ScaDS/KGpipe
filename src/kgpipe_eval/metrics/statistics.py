from kgpipe_eval.utils.kg_utils import TripleGraph
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

# @lru_cache(maxsize=1)
def count_measures(kg: TripleGraph) -> CountMeasures:
    
    triple_count = 0
    subject_count = 0 # TODO misses shallow object entities

    class_occurrence = defaultdict(int)
    property_occurrence = defaultdict(int)

    for _ in kg.subjects():
        subject_count += 1
    
    for s, p, o in kg.triples((None, None, None)):
        triple_count += 1
        if p == RDF.type:
            class_occurrence[str(o)] += 1
        property_occurrence[str(p)] += 1

    return CountMeasures(
        entity_count=subject_count,
        property_count=len(property_occurrence.keys()),
        triple_count=triple_count,
        class_count=len(class_occurrence.keys()),
        class_occurrence=class_occurrence,
        property_occurrence=property_occurrence,
    )

class CountMetric(Metric):
    key = "CountMetric"
    description = "Counts triples/classes/properties (basic statistics)."

    def compute(self, kg: TripleGraph) -> MetricResult:
        counts = count_measures(kg)
        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="entity_count", value=counts.entity_count, unit="number"),
                Measurement(name="triple_count", value=counts.triple_count, unit="number"),
                Measurement(name="property_count", value=counts.property_count, unit="number"),
                Measurement(name="class_count", value=counts.class_count, unit="number"),
                Measurement(name="property_occurrence", value=counts.property_occurrence, unit="dictionary"),
                Measurement(name="class_occurrence", value=counts.class_occurrence, unit="dictionary"),
            ],
            summary=f"Measures of entities, triples, properties, classes, property occurrences, and class occurrences"
        )

class DegreeMetric(Metric):
    # def compute(self, kg: TripleGraph) -> MetricResult:
    #     degrees = degree_measures(kg)
    #     return MetricResult(
    #         metric=self,
    #         measurements=[
    #             Measurement(name="degree", value=degrees.degree, unit="number"),
    #         ],
    #         summary=f"Measures of degrees"
    #     )
    pass