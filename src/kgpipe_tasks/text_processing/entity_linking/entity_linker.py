from sentence_transformers import util
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from kgpipe.common import KgTask, Data, DataFormat, Registry
from kgpipe.execution import docker_client

from typing import List, Tuple, Dict
from rdflib import Graph, RDFS
import os
from tqdm import tqdm
import json

from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document, TE_Pair

from kgpipe.util.embeddings import global_encode

class EntityMatch:
    def __init__(self, entity: str, label: str, score: float):
        self.entity = entity
        self.label = label
        self.score = score


class AliasAndLabelBasedEntityLinker():
    """
    Uses a transformer model to embed extracted relations and ontology predicates.
    Each predicate includes multiple aliases and one label.
    For matching, it computes cosine similarity between the extracted relation and
    each alias/label separately, then returns the predicate with the highest similarity.
    """
    def __init__(self, graph: Graph):
        self.graph = graph
        # get entity_uri label tuples
        entity_uri_label_tuples = []
        for s, _, l in self.graph.triples((None, RDFS.label, None)):
            entity_uri_label_tuples.append((s, l))
        self.entity_uri_label_tuples = entity_uri_label_tuples
        # self.model = SentenceTransformer("all-MiniLM-L6-v2")
        entity_texts = [f"{l}" for _, l in entity_uri_label_tuples]
        self.entity_embeddings = global_encode(entity_texts)

    def link_entities(self, extracted_entities: List[str]) -> List[EntityMatch]:
        if len(extracted_entities) == 0:
            # print("No extracted relations")
            return []

        # if extracted_relations in self.match_cache:
        #     return self.match_cache[extracted_relations]

        best_matches = []

        key_texts = extracted_entities
        key_embeddings = global_encode(key_texts)
        # key_embeddings = key_embeddings.to(self.entity_embeddings.device)


        if key_embeddings.shape[1] != self.entity_embeddings.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: {key_embeddings.shape[1]} vs {self.entity_embeddings.shape[1]}")

        similarities = util.cos_sim(key_embeddings, self.entity_embeddings)

        for i, key in enumerate(key_texts):
            best_idx = int(similarities[i].argmax())
            best_score = float(similarities[i][best_idx])
            match = self.entity_uri_label_tuples[best_idx]
            
            best_matches.append(EntityMatch(key, match[0], best_score))

        return best_matches
        

@Registry.task(
    input_spec={"source": DataFormat.TE_JSON, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.TE_JSON},
    description="Link entities using transformer embeddings on property label+alias TODO chain replacement",
    category=["TextProcessing", "EntityLinking"]
)
def label_alias_embedding_el(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    graph = Graph()
    graph.parse(inputs["target"].path, format="nt")
    linker = AliasAndLabelBasedEntityLinker(graph)
    
    if os.path.isdir(inputs["source"].path):
        os.makedirs(outputs["output"].path, exist_ok=True)
        for file in tqdm(os.listdir(inputs["source"].path), desc="Linking entities"):
            te_doc_in  = TE_Document(**json.load(open(os.path.join(inputs["source"].path, file))))
            entity_texts = list({triple.subject.surface_form for triple in te_doc_in.triples if triple.subject.surface_form})
            entity_texts += list({triple.object.surface_form for triple in te_doc_in.triples if triple.object.surface_form})
            entity_matches = linker.link_entities(entity_texts)
            te_links = [TE_Pair(span=match.entity, mapping=match.label, link_type="entity", score=match.score) for match in entity_matches]
            te_doc_out = te_doc_in.model_copy(deep=True)
            te_doc_out.links += te_links
            with open(os.path.join(outputs["output"].path, file), "w") as f:
                f.write(te_doc_out.model_dump_json())
    else:
        te_doc_in  = TE_Document(**json.load(open(inputs["source"].path)))
        entity_matches = linker.link_entities(list({triple.subject.surface_form for triple in te_doc_in.triples if triple.subject.surface_form}))
        te_links = [TE_Pair(span=match.entity, mapping=match.label, link_type="entity", score=match.score) for match in entity_matches]
        te_doc_out = te_doc_in.model_copy(deep=True)
        te_doc_out.links += te_links
        with open(outputs["output"].path, "w") as f:
            f.write(te_doc_out.model_dump_json())