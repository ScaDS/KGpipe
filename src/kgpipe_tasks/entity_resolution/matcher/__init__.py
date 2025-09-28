"""
Matcher tasks for entity linking and matching.
"""

from .paris_rdf_matcher import paris_exchange, paris_entity_matching
from .jedai_tab_matcher import pyjedai_entity_matching
# from .jedai_tab_matcher import pyjedai_entity_matching

# # Register tasks with the registry
# from kgpipe.common.registry import Registry

# def get_paris_entity_matching_task():
#     """Get the Paris entity matching task."""
#     return paris_entity_matching_task

# def get_paris_csv_to_matching_format_task():
#     """Get the Paris CSV to matching format task."""
#     return paris_csv_to_matching_format_task

# def get_jedai_tab_matcher_task():
#     """Get the Jedai tab matcher task."""
#     return jedai_tab_matcher_task

# # Register the task factory functions
# Registry.register("task")(get_paris_entity_matching_task)
# Registry.register("task")(get_paris_csv_to_matching_format_task)
# Registry.register("task")(get_jedai_tab_matcher_task)
# __all__ = [
#     "paris_entity_matching_task", 
#     "paris_csv_to_matching_format_task",
#     "get_paris_entity_matching_task",
#     "get_paris_csv_to_matching_format_task",
#     "get_jedai_tab_matcher_task"
# ] 