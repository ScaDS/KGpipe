from kgpipe.util.embeddings.st_emb import get_model
from kgpipe_eval.utils.alignment_utils import Alignment, align_entities_by_label_embedding
from kgpipe_eval.api import Metric, MetricResult, Measurement

from pydantic import BaseModel
from kgpipe.common import KG
import numpy as np

DEBUG = False

class DuplicateConfig(BaseModel):
    threshold: float = 0.5
    similarity_model: str = "cosine"
    similarity_function: str = "cosine"
    reference_kg: KG

def eval_duplicates(kg: KG, config: DuplicateConfig): 
    """
    checks expected & integrated source entity overlap using label embeddings
    """

    alignments : list[Alignment] = align_entities_by_label_embedding(kg, ref_kg, threshold=config.threshold, model=config.similarity_model, similarity=config.similarity_function)

    duplicates = 0
    already_matched_references = set()

    for alignment in alignments:
        if alignment.target in already_matched_references:
            duplicates += 1
        already_matched_references.add(alignment.target)

    return duplicates

class DuplicateMetric(Metric):
    def compute(self, kg: KG, ref_kg: KG, config: DuplicateConfig):
        duplicates = eval_duplicates(kg, ref_kg, config)
        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="duplicates", value=duplicates, unit="number"),
            ],
            summary=f"Duplicates in the KG"
        )

# find all duplicate entities in the KG
# using
# - reference KG
# - fuzzy matching
# - exact matching
# - semantic matching
# - clustering
# - ...
# return a list of duplicate entities
# return a list of duplicate entities with the matching score
# return a list of duplicate entities with the matching score and the matching type
# return a list of duplicate entities with the matching score and the matching type and the matching details
# return a list of duplicate entities with the matching score and the matching type and the matching details and the matching details
# return a list of duplicate entities with the matching score and the matching type and the matching details and the matching details and the matching details