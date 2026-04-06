from kgpipe.common import KG
from typing import Literal, NamedTuple
from functools import lru_cache
from pydantic import BaseModel

from kgpipe_eval.utils.kg_utils import TripleGraph, Term, Triple
from kgpipe.util.embeddings.st_emb import get_model

from rdflib import RDFS, RDF
import numpy as np

class AlignmentConfig(BaseModel):
    model: str = "sentence-transformer"
    similarity: str = "cosine"
    threshold: float = 0.5

# TODO source entities csv to label only graph

CONFIG=None
# layz config dict
def get_config() -> dict:
    global CONFIG
    if CONFIG is None:
        # TODO
        pass
    return CONFIG

EntityAlignment = NamedTuple("EntityAlignment", [("source", Term), ("target", Term)])
TripleAlignment = NamedTuple("TripleAlignment", [("source", Triple), ("target", Triple)])

# Core alignment method interfaces

@lru_cache(maxsize=1000)
def get_aligned_entities(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[Entity]:
    return kg.entities.intersection(reference_kg.entities)

def get_aligned_triples(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[Triple]:
    return kg.triples.intersection(reference_kg.triples)

# Helper methods

def get_entity_uri_label_pairs(triple_graph: TripleGraph) -> list[tuple[Term, Term]]:
    return [(s, label) for s, _, label in triple_graph.triples((None, RDFS.label, None))]

# Specific alignment methods

def align_entities_by_label_embedding(triple_graph: TripleGraph, ref_triple_graph: TripleGraph, model="TODO", similarity="cosine", threshold=0.5):
    model = get_model()
    ref_entity_labels = [str(label) for s, _, label in ref_triple_graph.triples((None, RDFS.label, None))]
    ref_entity_labels_embeddings = model.encode(ref_entity_labels, convert_to_numpy=True, show_progress_bar=False)

    gen_entity_labels = [str(label) for s, _, label in triple_graph.triples((None, RDFS.label, None))]
    gen_entity_labels_embeddings = model.encode(gen_entity_labels, convert_to_numpy=True, show_progress_bar=False)

    for s, _, label in triple_graph.triples((None, RDFS.label, None)):
        gen_entity_labels.append(str(label))

    sims = np.dot(gen_entity_labels_embeddings, ref_entity_labels_embeddings.T)

    alignments = []
    for i in range(sims.shape[0]):
        best_j = np.argmax(sims[i])
        if sims[i][best_j] >= threshold:
            alignments.append(EntityAlignment(source=gen_entity_labels[i], target=ref_entity_labels[best_j], score=sims[i][best_j]))
    return alignments

def align_by_label_alias_embedding(triple_graph: TripleGraph, model="", similarity="cosine", threshold=0.5):
    pass
