"""Public package surface.

``Pipeline`` and the types around it load on first use.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CatalogValidationError",
    "KgPipePlan",
    "Pipeline",
    "PipelineConf",
    "PipelineRun",
    "PipelineTaskConf",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module 'kgpipe' has no attribute {name!r}")
    from kgpipe import api

    return getattr(api, name)
