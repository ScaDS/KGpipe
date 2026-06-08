from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from kgpipe.common.model.kg import KG

# TODO move parts from kgpipe.evaluation.base to here

class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str, description: str | None = None):
        self.name = name
        self.description = description or name
    
    @abstractmethod
    def compute(self, ground_truth: KG, prediction: KG) -> float:
        """Compute the metric value."""
        pass
    
    def __str__(self) -> str:
        return f"Metric({self.name})"


@dataclass
class EvaluationReport:
    """Contains the results of evaluating a KG against a ground truth."""
    id: str
    ground_truth: KG
    prediction: KG
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4().hex)
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric result to the report."""
        self.metrics[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            "id": self.id,
            "ground_truth": self.ground_truth.name,
            "prediction": self.prediction.name,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        return f"EvaluationReport({self.ground_truth.name} vs {self.prediction.name}, metrics={len(self.metrics)})"
