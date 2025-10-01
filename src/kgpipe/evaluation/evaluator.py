"""
Main Evaluator Module

Orchestrates evaluation across all three aspects: statistical, semantic, and reference-based.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..common.models import KG, Data
from .base import EvaluationAspect, AspectResult, EvaluationConfig
from .metrics import MetricResult
from .reports import EvaluationReport


class Evaluator:
    """Main evaluator that orchestrates evaluation across all aspects."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.aspect_evaluators = self._initialize_aspect_evaluators()
    
    def _initialize_aspect_evaluators(self) -> Dict[EvaluationAspect, Any]:
        """Initialize evaluators for each aspect."""
        evaluators = {}
        
        for aspect in self.config.aspects:
            if aspect == EvaluationAspect.STATISTICAL:
                from .aspects.statistical import StatisticalEvaluator
                evaluators[aspect] = StatisticalEvaluator()
            elif aspect == EvaluationAspect.SEMANTIC:
                from .aspects.semantic import SemanticEvaluator
                evaluators[aspect] = SemanticEvaluator()
            elif aspect == EvaluationAspect.REFERENCE:
                from .aspects.reference import ReferenceEvaluator
                evaluators[aspect] = ReferenceEvaluator()
        
        return evaluators
    
    def evaluate(self, kg: KG, references: Dict[str, Data] = {}) -> EvaluationReport:
        """Evaluate the KG across all configured aspects."""
        if not kg.exists():
            raise FileNotFoundError(f"KG file not found: {kg.path}")
        
        if references is {}:
            raise ValueError("References are required for reference-based evaluation")
        
        aspect_results = []
        all_metrics = []
        
        # Evaluate each aspect
        for aspect in self.config.aspects:
            if aspect in self.aspect_evaluators:
                evaluator = self.aspect_evaluators[aspect]
                
                # Prepare kwargs for aspect evaluation
                kwargs = {}
                if aspect == EvaluationAspect.REFERENCE:
                    kwargs['references'] = references
                
                if self.config.metrics:
                    kwargs['metrics'] = self.config.metrics
                
                try:
                    aspect_result = evaluator.evaluate(kg, **kwargs)
                    aspect_results.append(aspect_result)
                    all_metrics.extend(aspect_result.metrics)
                except Exception as e:
                    # Log error but continue with other aspects
                    print(f"Warning: Failed to evaluate {aspect.value} aspect: {e}")
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(aspect_results)
        
        # Create evaluation report
        report = EvaluationReport(
            kg=kg,
            references=references,
            aspect_results=aspect_results,
            overall_score=overall_score,
            config=self.config
        )
        
        return report
    
    def evaluate_aspect(self, aspect: EvaluationAspect, kg: KG, **kwargs) -> AspectResult:
        """Evaluate a specific aspect of the KG."""
        if aspect not in self.aspect_evaluators:
            raise ValueError(f"No evaluator available for aspect: {aspect}")
        
        evaluator = self.aspect_evaluators[aspect]
        return evaluator.evaluate(kg, **kwargs)
    
    def get_available_metrics(self, aspect: Optional[EvaluationAspect] = None) -> List[str]:
        """Get available metrics for specified aspect or all aspects."""
        if aspect:
            if aspect not in self.aspect_evaluators:
                return []
            return self.aspect_evaluators[aspect].get_available_metrics()
        
        all_metrics = []
        for evaluator in self.aspect_evaluators.values():
            all_metrics.extend(evaluator.get_available_metrics())
        return all_metrics
    
    def _calculate_overall_score(self, aspect_results: List[AspectResult]) -> float:
        """Calculate overall score from aspect results."""
        if not aspect_results:
            return 0.0
        
        # Simple average for now, can be made configurable
        total_score = sum(result.overall_score for result in aspect_results)
        return total_score / len(aspect_results)

