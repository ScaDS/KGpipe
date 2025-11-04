from .simple import aggregate_2matches, reduce_to_best_match_per_entity
from .preference import fusion_first_value, select_first_value
from .majority import majority_fusion

__all__ = ["aggregate_2matches", "reduce_to_best_match_per_entity", "fusion_first_value", "majority_fusion", "select_first_value"]