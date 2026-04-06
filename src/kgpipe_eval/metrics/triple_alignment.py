from pydantic import BaseModel
from typing import Literal

from kgpipe.common import KG
from kgpipe_eval.utils.alignment_utils import Alignment

# measures precision, recall, f1 score, etc.

class TripleAlignmentConfig(BaseModel):
    reference_kg: KG
    similarity_threshold: float = 0.5
    similarity_model: str = "cosine"
    similarity_function: str = "cosine"

def eval_triple_alignment(method: Literal["exact", "fuzzy", "semantic"] = "exact"):
    pass

def eval_triple_alignment_by_label_embedding(method: Literal["exact", "fuzzy", "semantic"] = "exact"):
    pass


class ReferenceTripleAlignmentMetric(Metric):
    pass