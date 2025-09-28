"""
Base classes for evaluation system.

This module contains the core classes to avoid circular imports.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..common.models import KG


class EvaluationAspect(Enum):
    """The three main aspects of KG evaluation."""
    STATISTICAL = "statistical"
    SEMANTIC = "semantic"
    REFERENCE = "reference"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    aspects: List[EvaluationAspect] = field(default_factory=lambda: list(EvaluationAspect))
    metrics: Optional[List[str]] = None  # Specific metrics to compute
    weights: Optional[Dict[str, float]] = None  # Weights for different metrics
    reference_kg: Optional[KG] = None
    output_format: str = "json"
    include_details: bool = True
    generate_report: bool = True
    
    def __post_init__(self):
        if self.weights and not all(0.0 <= w <= 1.0 for w in self.weights.values()):
            raise ValueError("All weights must be between 0.0 and 1.0")


@dataclass
class AspectResult:
    """Result of evaluating a specific aspect."""
    aspect: EvaluationAspect
    metrics: List["MetricResult"]
    overall_score: float
    details: Dict[str, Any]
    
    def __post_init__(self):
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError("Overall score must be between 0.0 and 1.0")

    def __str__(self) -> str:
        return f"{self.aspect.value}: {self.overall_score:.2f}"


class AspectEvaluator(ABC):
    """Base class for aspect-specific evaluators."""
    
    def __init__(self, aspect: EvaluationAspect):
        self.aspect = aspect
    
    @abstractmethod
    def evaluate(self, kg: KG, **kwargs) -> AspectResult:
        """Evaluate the KG for this specific aspect."""
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics for this aspect."""
        pass


@dataclass
class MetricResult:
    """Result of computing a single metric."""
    name: str
    value: float
    normalized_score: float  # 0.0-1.0 range
    details: Dict[str, Any]
    aspect: EvaluationAspect
    duration: float = 0.0
    
    def __post_init__(self):
        if not 0.0 <= self.normalized_score <= 1.0:
            raise ValueError("Normalized score must be between 0.0 and 1.0")

class MetricConfig(BaseModel):
    name: str

class Metric(ABC):
    """Base class for all evaluation metrics."""
    
    def __init__(self, name: str, description: str, aspect: EvaluationAspect, metricConfig: Optional[MetricConfig] = None):
        self.name = name
        self.description = description
        self.aspect = aspect
        self.metricConfig = metricConfig
    
    @abstractmethod
    def compute(self, kg, **kwargs) -> MetricResult:
        """Compute the metric value for the given KG."""
        pass
    
    def normalize_score(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a value to 0-1 range."""
        if max_val == min_val:
            return 0.5  # Default to middle value if range is zero
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def __str__(self) -> str:
        return f"{self.name} ({self.aspect.value})"
    
    def __repr__(self) -> str:
        return f"Metric(name='{self.name}', aspect={self.aspect.value})" 