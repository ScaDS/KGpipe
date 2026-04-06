from kgpipe_eval.api import Metric

from pydantic import BaseModel
from kgpipe.common import KG

class DisjointDomainConfig(BaseModel):
    pass

class DomainConfig(BaseModel):
    pass

class RangeConfig(BaseModel):
    pass

class RelationDirectionConfig(BaseModel):
    pass

class DisjointDomainMetric(Metric):
    def compute(self, kg: KG, ref_kg: KG, config: DisjointDomainConfig):
        pass

class DomainMetric(Metric):
    pass

class RangeMetric(Metric):
    pass

class RelationDirectionMetric(Metric):
    pass

class DatatypeMetric(Metric):
    pass

class DatatypeFormatMetric(Metric):
    pass

# class OntologyClassCoverageMetric():
#     pass

# class OntologyRelationCoverageMetric():
#     pass

# class OntologyNamespaceCoverageMetric():
#     pass