from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class PipelineFamily(str, Enum):
    RDF = "rdf"
    TEXT = "text"


class SearchMethod(str, Enum):
    DEFAULT = "default"
    RANDOM = "random"
    QUALITY_AWARE = "quality_aware"


class SearchSpaceMode(str, Enum):
    JOINT = "joint"
    IMPLEMENTATION_ONLY = "implementation_only"
    PARAMETER_ONLY = "parameter_only"


@dataclass(frozen=True)
class PipelineConfig:
    family: PipelineFamily
    implementations: Mapping[str, str]
    params: Mapping[str, float]

    def as_dict(self) -> dict:
        return {
            "family": self.family.value,
            "implementations": dict(self.implementations),
            "params": dict(self.params),
        }

