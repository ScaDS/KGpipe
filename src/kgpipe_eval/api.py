from dataclasses import dataclass
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Callable
from kgpipe.common.model.kg import KgKg


# MetricConfig (rich, typed, input)
#         ↓
# computation
#         ↓
# MetricResult
#     ├── measurements (results)
#     └── metadata (flattened config + context)

@dataclass(frozen=True)
class MetricConfig:
    pass

@dataclass(frozen=True)
class Measurement:
    name: str
    value: int | float | str | bool
    unit: str | None = None

@dataclass(frozen=True)
class MetricResult:
    metric: "Metric"
    measurements: list[Measurement]
    summary: str | None = None
    # TODO metadata/properties: dict[str, int | float | str | bool] = field(default_factory=dict)

@dataclass(frozen=True)
class Metric:
    key: str
    description: str
    compute: Callable[[KgKg, MetricConfig], MetricResult]


# ---

