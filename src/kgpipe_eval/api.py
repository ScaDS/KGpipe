from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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
    value: Any
    unit: str | None = None

@dataclass(frozen=True)
class MetricResult:
    metric: "Metric"
    measurements: list[Measurement]
    summary: str | None = None
    # TODO metadata/properties: dict[str, int | float | str | bool] = field(default_factory=dict)

class Metric(ABC):
    """
    Minimal metric interface for the `kgpipe eval-new` CLI.

    Metrics are instantiated (usually with default config) and then run via `compute(...)`.
    """

    key: str
    description: str

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult: ...


# ---

