from kg_sge.api.correctness import SourceGroundCorrectenss, SourceGroundCorrectnessConfig
from kg_sge.api.coverage import SourceGroundedCoverage, SourceGroundedCoverageConfig
from kgpipe_eval.utils.kg_utils import KgManager, KG, KgLike

class SourceGroundedCorrectnessMetric:
    def __init__(self):
        self.correctness = SourceGroundCorrectenss()

    def compute(self, kg: KG, config: SourceGroundCorrectnessConfig):
        pass

class SourceGroundedCoverageMetric:
    def __init__(self):
        self.coverage = SourceGroundedCoverage()

    def compute(self, kg: KG, config: SourceGroundedCoverageConfig):
        pass