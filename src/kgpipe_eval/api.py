from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


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
class MeasurementSpec:
    """Declared output of a metric (schema only; no runtime value)."""
    name: str
    unit: str | None = None
    alias: tuple[str, ...] = ()

@dataclass(frozen=True)
class Measurement:
    name: str
    value: Any
    unit: str | None = None
    alias: tuple[str, ...] = ()

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
    Subclasses declare `key`, `description`, and `measurements` (output schema).
    """

    key: str
    description: str
    measurements: ClassVar[tuple[MeasurementSpec, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "key" not in cls.__dict__:
            cls.key = cls.__name__
        if "description" not in cls.__dict__:
            doc = (cls.__doc__ or "").strip()
            cls.description = doc.split("\n", 1)[0] if doc else cls.__name__

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult: ...


# ---
