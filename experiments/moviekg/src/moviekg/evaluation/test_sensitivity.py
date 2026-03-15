from dataclasses import dataclass
from typing import List
from kgpipe.evaluation.aspects.reference import ReferenceEvaluator, ReferenceConfig

@dataclass
class BinaryClassifier:
    tp: int
    fp: int
    tn: int
    fn: int

    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)

@dataclass
class ThresholdSensitivityResult:
    pipeline_name: str
    threshold: float
    result: BinaryClassifier



reference_evaluator = ReferenceEvaluator()

def paris_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    ReferenceConfig(
        ENTITY_MATCH_THRESHOLD=threshold
        RELATION_MATCH_THRESHOLD=threshold        
    ) # TODO get config from dataset
    # kg = KG(path=Path(f"data/moviekg/paris/{pipeline_name}.nt"))
    # reference_kg = KG(path=Path("data/moviekg/paris/reference.nt"))
    # result = reference_evaluator.evaluate(kg, reference_kg)
    # return result
    pass

def jedai_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def valentine_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def corenlp_openie_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def dbpedia_spotlight_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def custom_relation_linking_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass

def custom_entity_linking_threshold_sensitivity(pipeline_name: str, threshold: float) -> List[ThresholdSensitivityResult]:
    pass