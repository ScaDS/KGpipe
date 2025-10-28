from kgpipe.evaluation.cluster import MatchCluster
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document
from .simple import aggregate_2matches, reduce_to_best_match_per_entity
from .preference import fusion_first_value
from kgpipe.common.registry import Registry
from typing import Optional
import json

# __all__ = ["get_union_rdf_task", "get_union_matched_rdf_task", "get_union_matched_rdf_combined_task", "aggregate_2matches"]
