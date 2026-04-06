from kgpipe_eval.utils.kg_utils import Term
from typing import NamedTuple
from kgpipe.util.embeddings.st_emb import get_model
from kgpipe.common import KG
from typing import Literal
from kgpipe_eval.utils.measurement_utils import BCMeasurement

from kgpipe_eval.api import Metric
from pydantic import BaseModel
from rdflib import RDFS

# TODO
# measures precision, recall, f1 score, etc.

TODO = None

def eval_entity_alignment(kg: KG, config: TODO):
    pass

def eval_entity_alignment_by_label_embedding(kg: KG, threshold: float = 0.5):
    model = get_model()
    # entity_dict = load_entity_dict(entity_dict_path)
    # entity_labels = list(set([entity_dict[uri].entity_label for uri in entity_dict if entity_dict[uri].entity_label is not None]))
    entity_labels = [] # TODO: get entity labels from the KG
    entity_labels_embeddings = model.encode(entity_labels, convert_to_numpy=True, show_progress_bar=False)

    found_labels = []
    overlapping_entities = set()
    overlapping_entities_strict = set()

    graph = kg.get_graph()
    for s, p, label in graph.triples((None, RDFS.label, None)):
        found_labels.append(str(label))

    found_labels_embeddings = model.encode(found_labels, convert_to_numpy=True, show_progress_bar=False)

    return BCMeasurement(
        tp=len(overlapping_entities),
        fp=len(found_labels) - len(overlapping_entities),
        tn=len(kg.entities) - len(found_labels),
        fn=len(kg.entities) - len(overlapping_entities)
    )

class EntityAlignmentMetric(Metric):
    pass