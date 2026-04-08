from kgpipe.common import KG
from typing import Literal, NamedTuple, Optional
from functools import lru_cache
from pydantic import BaseModel, ConfigDict, model_validator

from kgpipe_eval.utils.kg_utils import TripleGraph, Term, Triple
from kgpipe.util.embeddings.st_emb import get_model

from rdflib import RDFS, RDF
from kgpipe.datasets.multipart_multisource import read_entities_csv, EntitiesRow
import numpy as np
from pathlib import Path

# TODO source entities csv to label only graph

class EntityAlignmentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    method: Literal["label_embedding", "label_alias_embedding", "label_embedding_and_type"] = "label_embedding"
    reference_kg: Optional[KG] = None
    verified_entities_path: Optional[Path] = None
    verified_entities_delimiter: str = "\t"
    entity_sim_threshold: float = 0.95

    # value_sim_threshold: float = 0.5

    @model_validator(mode="after")
    def _require_reference_source(self):
        if self.reference_kg is None and self.verified_entities_path is None:
            raise ValueError("Provide either `reference_kg` or `verified_entities_path`.")
        return self


EntityAlignment = NamedTuple("EntityAlignment", [("source", Term), ("target", Term), ("score", float)])
TripleAlignment = NamedTuple("TripleAlignment", [("source", Triple), ("target", Triple)])

# Core alignment method interfaces

@lru_cache(maxsize=1000)
def get_aligned_entities(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[EntityAlignment]:
    return kg.entities.intersection(reference_kg.entities)

def get_aligned_triples(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[TripleAlignment]:
    return kg.triples.intersection(reference_kg.triples)

# Helper methods

# def get_entity_uri_label_pairs(triple_graph: TripleGraph) -> list[tuple[Term, Term]]:
#     return [(s, label) for s, _, label in triple_graph.triples((None, RDFS.label, None))]

UriLabelTypePair = NamedTuple("UriLabelTypePair", [("uri", Term), ("label", Term), ("type", Term)])

def get_entity_uri_label_type_pairs(kg: KG) -> list[UriLabelTypePair]:
    label_by_uri = {}
    type_by_uri = {}
    for s, p, o in kg.triples((None, RDFS.label, None)):
        label_by_uri[str(s)] = str(o)
    for s, p, o in kg.triples((None, RDF.type, None)):
        type_by_uri[str(s)] = str(o)
    for uri in label_by_uri:
        if uri in type_by_uri:
            yield UriLabelTypePair(uri=uri, label=label_by_uri[uri], type=type_by_uri[uri])
        else:
            yield UriLabelTypePair(uri=uri, label=label_by_uri[uri], type=None)

def load_verified_entities(path: Path, delimiter: str = "\t") -> list[UriLabelTypePair]:
    """
    """
    if path.name.endswith(".json"):
        raise ValueError("JSON format not supported for verified entities")
    elif path.name.endswith(".csv"):
        return [UriLabelTypePair(uri=entity.entity_id, label=entity.entity_label, type=entity.entity_type) for entity in read_entities_csv(path=path, delimiter=delimiter)]
    else:
        raise ValueError(f"Unsupported file type: {path}")

def load_entity_uri_label_type_pairs(config: EntityAlignmentConfig) -> list[UriLabelTypePair]:
    if config.verified_entities_path is not None:
        return load_verified_entities(config.verified_entities_path, delimiter=config.verified_entities_delimiter)
    elif config.reference_kg is not None:
        return get_entity_uri_label_type_pairs(config.reference_kg)
    else:
        raise ValueError("No verified entities path or reference KG provided")

# Specific alignment methods

def align_entities_by_label_embedding(tg: TripleGraph, config: EntityAlignmentConfig) -> list[EntityAlignment]:
    model = get_model()
    ref_entity_uri_label_type_pairs = load_entity_uri_label_type_pairs(config)
    ref_labels = [pair.label for pair in ref_entity_uri_label_type_pairs]
    ref_labels_embeddings = model.encode(ref_labels, convert_to_numpy=True, show_progress_bar=False)

    gen_entity_uri_label_type_pairs = list(get_entity_uri_label_type_pairs(tg))
    gen_labels = [pair.label for pair in gen_entity_uri_label_type_pairs]
    gen_labels_embeddings = model.encode(gen_labels, convert_to_numpy=True, show_progress_bar=False)


    sims = np.dot(gen_labels_embeddings, ref_labels_embeddings.T)

    alignments = []
    for i in range(sims.shape[0]):
        best_j = np.argmax(sims[i])
        if sims[i][best_j] >= config.entity_sim_threshold:
            alignments.append(EntityAlignment(source=gen_entity_uri_label_type_pairs[i].uri, target=ref_entity_uri_label_type_pairs[best_j].uri, score=sims[i][best_j]))
    return alignments

def align_by_label_alias_embedding(triple_graph: TripleGraph, model="", similarity="cosine", threshold=0.5):
    pass
