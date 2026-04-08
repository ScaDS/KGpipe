from pydantic import BaseModel, ConfigDict
from typing import Literal

from kgpipe.common import KG
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentConfig
from kgpipe_eval.utils.measurement_utils import BCMeasurement
from kgpipe_eval.api import Metric

# measures precision, recall, f1 score, etc.

class TripleAlignmentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reference_kg: KG
    entity_alignment_config: EntityAlignmentConfig
    value_sim_threshold: float = 0.5

# def eval_triple_alignment(method: Literal["exact", "fuzzy", "semantic"] = "exact"):
#     pass

def eval_triple_alignment_by_label_embedding(method: Literal["exact", "fuzzy", "semantic"] = "exact"):
    pass


def eval_triple_alignment_by_label_embedding_soft_literals(method: Literal["exact", "fuzzy", "semantic"] = "exact"):
    pass

class ReferenceTripleAlignmentMetric(Metric):
    pass


# Backward-compatibility alias (imported by `kgpipe_eval.metrics.__init__`).
TripleAlignmentMetric = ReferenceTripleAlignmentMetric