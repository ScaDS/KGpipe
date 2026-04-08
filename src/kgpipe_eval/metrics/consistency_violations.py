from kgpipe_eval.api import Metric

from pydantic import BaseModel, model_validator, ConfigDict
from kgpipe.common import KG
from pathlib import Path
from kgpipe_eval.utils.kg_utils import TripleGraph

class ConsistencyViolationsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reference_kg: KG
    ontology_path: Path

    @model_validator(mode="after")
    def _require_reference_kg_or_ontology_path(self):
        if self.reference_kg is None and self.ontology_path is None:
            raise ValueError("Provide either `reference_kg` or `ontology_path`.")
        return self

class DisjointDomainMetric(Metric):
    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        pass

class DomainMetric(Metric):
    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        pass

class RangeMetric(Metric):
    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        pass

class RelationDirectionMetric(Metric):
    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        pass

class DatatypeMetric(Metric):
    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        pass

class DatatypeFormatMetric(Metric):
    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        pass

# class OntologyClassCoverageMetric():
#     pass

# class OntologyRelationCoverageMetric():
#     pass

# class OntologyNamespaceCoverageMetric():
#     pass