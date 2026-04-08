from kgpipe_eval.utils.alignment_utils import EntityAlignment, align_entities_by_label_embedding, EntityAlignmentConfig
from kgpipe_eval.api import Metric, MetricResult, Measurement
from kgpipe_eval.utils.kg_utils import Term, TripleGraph

from pydantic import BaseModel, ConfigDict
from kgpipe.common import KG
import numpy as np

DEBUG = False

class DuplicateConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_alignment_config: EntityAlignmentConfig

class DuplicateMeasures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    duplicates: int
    total_references: int
    already_matched_references: set[Term]

def eval_duplicates(kg: TripleGraph, config: DuplicateConfig): 
    """
    checks expected & integrated source entity overlap using label embeddings
    """

    alignments : list[EntityAlignment] = align_entities_by_label_embedding(kg, config.entity_alignment_config)

    duplicates = set()
    already_matched_references = set()

    for alignment in alignments:
        if alignment.target in already_matched_references:
            duplicates.add(alignment.target)
        already_matched_references.add(alignment.target)

    if DEBUG:
        print("Duplicates:")
        for alignment in alignments:
            if alignment.target in duplicates:
                print(alignment.target, alignment.source, alignment.score)

    return duplicates

class DuplicateMetric(Metric):
    def compute(self, kg: TripleGraph, config: DuplicateConfig):
        duplicates = eval_duplicates(kg, config)
        entity_count = len(list(kg.entities()))
        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="duplicates", value=len(duplicates), unit="number"),
                Measurement(name="entity_count", value=entity_count, unit="number"),
                Measurement(name="duplicates_ratio", value=len(duplicates) / entity_count, unit="percentage"),
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