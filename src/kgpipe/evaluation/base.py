"""
Base classes for evaluation system.

This module contains the core classes to avoid circular imports.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
# from kgpipe.common.systemgraph import kg_class
from kgpipe.common.systemgraph import PipeKG
import time
import json
import functools
import inspect
# from kgpipe.common.util import create_insertable_nodes_and_edges, insert_kg_obj
from pydantic import BaseModel

from kgpipe.common.models import KG
from kgpipe.common.definitions import MetricEntity, MetricRunEntity, MetricEntityId, DataHandle
from kgpipe.common.config import config
from pathlib import Path
from kgpipe.common.util import encode_string
class EvaluationAspect(Enum):
    """The three main aspects of KG evaluation."""
    STATISTICAL = "statistical"
    SEMANTIC = "semantic"
    REFERENCE = "reference"
    SPECIFIC = "specific"


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
    metric_config_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.weights and not all(0.0 <= w <= 1.0 for w in self.weights.values()):
            raise ValueError("All weights must be between 0.0 and 1.0")

    # def get_aspect_config(self, aspect: EvaluationAspect) -> MetricConfig:
    #     if aspect == EvaluationAspect.STATISTICAL:
    #         return StatisticalConfig(name="default")
    #     elif aspect == EvaluationAspect.SEMANTIC:
    #         return SemanticConfig(name="default")
    #     elif aspect == EvaluationAspect.REFERENCE:
    #         return ReferenceConfig(name="default")
    #     else:
    #         raise ValueError(f"No config available for aspect: {aspect}")

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




# @Track(with_timestamp=True)
# @kg_class(type="MetricResult", description="Result of computing a single metric.")
class MetricResult(BaseModel):
    """Result of computing a single metric."""
    name: str
    value: float
    normalized_score: float  # 0.0-1.0 range
    details: Dict[str, Any]
    aspect: EvaluationAspect
    duration: float = 0.0
    input: str = "" # TODO
    
    def __post_init__(self):
        if not 0.0 <= self.normalized_score <= 1.0:
            raise ValueError("Normalized score must be between 0.0 and 1.0")

class MetricConfig(BaseModel):
    name: str

class AspectEvaluator(ABC):
    """Base class for aspect-specific evaluators."""
    
    def __init__(self, aspect: EvaluationAspect):
        self.aspect = aspect
    
    @abstractmethod
    def evaluate(self, kg: KG, config: Optional[MetricConfig], **kwargs) -> AspectResult:
        """Evaluate the KG for this specific aspect."""
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics for this aspect."""
        pass


def save_metric_run(metric: MetricResult):

    metric_run_entity = MetricRunEntity(
        status="success",
        started_at=time.time(),
        ended_at=time.time(),
        computedMetric=MetricEntityId(config.PIPEKG_PREFIX+encode_string(metric.name)),
        input=[DataHandle(uri=metric.input, type="any/text")],
        value=metric.value, 
        details=json.dumps(metric.details, default=str)
    )
    PipeKG.add_metric_run(metric_run_entity)

    #     # input=metric.input,
    #     # output=metric.output
    # )

def track_metric_compute(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        started = time.perf_counter()

        try:
            bound = inspect.signature(func).bind(self, *args, **kwargs)
            bound.apply_defaults()
            kg = bound.arguments.get("kg")
            config = bound.arguments.get("config")
        except Exception:
            kg: KG = None
            config = None

        try:
            result = func(self, *args, **kwargs)  # <-- actually call it
            result.input = str(kg.path)
            save_metric_run(result)
            return result
        except Exception as e:
            raise
    return wrapper

# def _safe_summarize_result(result):
#     """Return a JSON-serializable, compact view of the result."""
#     try:
#         # If your MetricResult is pydantic/dataclass, adapt this.
#         return result.to_dict() if hasattr(result, "to_dict") else str(result)
#     except Exception:
#         return str(result)

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
    
    def __str__(self) -> str:
        return f"{self.name} ({self.aspect.value})"
    
    def __repr__(self) -> str:
        return f"Metric(name='{self.name}', aspect={self.aspect.value})" 

    def __init_subclass__(cls, **kwargs):
        """Automatically wrap concrete compute implementations with tracking."""
        super().__init_subclass__(**kwargs)
        # Only wrap if this subclass overrides compute
        if 'compute' in cls.__dict__:
            #cls.compute = track_metric_compute(cls.__dict__['compute'])
            cls.compute = track_metric_compute(cls.__dict__['compute'])

    def normalize_score(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a value to 0-1 range."""
        if max_val == min_val:
            return 0.5  # Default to middle value if range is zero
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


