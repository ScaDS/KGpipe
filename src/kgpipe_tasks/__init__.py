"""
KGbench Extras

This package contains additional tools and formats for the KGbench framework.
"""

from kgpipe.common.registry import Registry

# Import new modules
from . import cleaning, common, construction, entity_resolution, reasoning_inference, schema_alignment, text_processing, transform_interop

# # Register tasks
# def get_paris_entity_matching_task():
#     return paris_entity_matching_task

# Registry.register("task")(get_paris_entity_matching_task)

__all__ = [
    "cleaning",
    "common",
    "construction",
    "entity_resolution",
    "reasoning_inference",
    "schema_alignment",
    "text_processing",
    "transform_interop"
]