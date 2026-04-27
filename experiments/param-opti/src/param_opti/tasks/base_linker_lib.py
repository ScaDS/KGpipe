import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from kgcore.api.ontology import OntologyUtil, OwlProperty
from kgpipe.common import Data, DataFormat, Registry
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document, TE_Pair
from rdflib import Graph, RDFS
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

_models: Dict[str, SentenceTransformer] = {}

class Embedder(ABC):
    def __init__(self, embedder_name: str):
        self.embedder_name = embedder_name
    
    @abstractmethod
    def encode_as_dict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass


def get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _models:
        model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            model.to(torch.cuda.current_device())
        _models[model_name] = model
    return _models[model_name]

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        super().__init__("sentence-transformer")
        self.model_name = model_name

    def encode_as_dict(self, text_list: List[str]) -> Dict[str, np.ndarray]:
        embeddings = self.encode(text_list)
        return {text: embedding for text, embedding in zip(text_list, embeddings)}

    def encode(self, text_list: List[str]) -> np.ndarray:
        embeddings = get_model(self.model_name).encode(text_list, show_progress_bar=False)
        return embeddings

class EntityMatch:
    def __init__(self, entity: str, label: str, score: float):
        self.entity = entity
        self.label = label
        self.score = score


def _validate_embedding_dimensions(query_embeddings: np.ndarray, target_embeddings: np.ndarray) -> None:
    if query_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"{query_embeddings.shape[1]} vs {target_embeddings.shape[1]}"
        )


class AliasAndLabelBasedEntityLinker:
    """
    Link extracted entity mentions to graph resources using label embeddings.
    """

    def __init__(self, graph: Graph, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.0):
        self.graph = graph
        self.embedder = SentenceTransformerEmbedder(model_name=model_name)
        self.threshold = float(threshold)
        self.entity_uri_label_tuples = [
            (entity_uri, str(label))
            for entity_uri, _, label in self.graph.triples((None, RDFS.label, None))
        ]
        entity_texts = [label for _, label in self.entity_uri_label_tuples]
        self.entity_embeddings = self.embedder.encode(entity_texts)

    def link_entities(self, extracted_entities: List[str]) -> List[EntityMatch]:
        if not extracted_entities:
            return []

        best_matches = []
        key_embeddings = self.embedder.encode(extracted_entities)
        _validate_embedding_dimensions(key_embeddings, self.entity_embeddings)
        similarities = util.cos_sim(key_embeddings, self.entity_embeddings)

        for i, entity in enumerate(extracted_entities):
            best_idx = int(similarities[i].argmax())
            best_score = float(similarities[i][best_idx])
            if best_score < self.threshold:
                continue
            entity_uri, _ = self.entity_uri_label_tuples[best_idx]
            best_matches.append(EntityMatch(entity, entity_uri, best_score))

        return best_matches
        


def label_alias_embedding_el(inputs: Dict[str, Data], outputs: Dict[str, Data], model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.5):
    graph = Graph()
    graph.parse(inputs["target"].path, format="nt")
    linker = AliasAndLabelBasedEntityLinker(graph, model_name=model_name, threshold=threshold)
    
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



class RelationMatch:
    def __init__(self, relation: str, predicate: OwlProperty, score: float):
        self.relation = relation
        self.predicate = predicate
        self.score = score

    def __str__(self):
        return f"RelationMatch(relation={self.relation}, predicate={self.predicate.uri}, score={self.score})"


def normalize(text):
    return text.replace('_', ' ').replace('-', ' ').strip().lower()

def build_property_text(prop: OwlProperty):
    text_parts = [
        f"label: {normalize(prop.label)}",
        f"altLabels: {', '.join(normalize(lbl) for lbl in prop.alias)}"
        # f"domain: {normalize(prop.get('domain', ''))}",
        # f"comment: {normalize(prop.get('comment', ''))}"
    ]
    return "; ".join(text_parts)

class AliasAndTransformerBasedRelationLinker:
    """
    Link extracted relation phrases to ontology predicates using label and alias embeddings.
    """

    def __init__(self, ontology_file, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.0):
        self.ontology = OntologyUtil.load_ontology_from_file(ontology_file)
        self.embedder = SentenceTransformerEmbedder(model_name=model_name)
        self.threshold = float(threshold)
        property_texts = [build_property_text(p) for p in self.ontology.properties]
        self.property_embeddings = self.embedder.encode(property_texts)

    def link_relations(self, extracted_relations: List[str]) -> List[RelationMatch]:
        if not extracted_relations:
            return []

        best_matches = []
        key_texts = [normalize(relation) for relation in extracted_relations]
        key_embeddings = self.embedder.encode(key_texts)
        _validate_embedding_dimensions(key_embeddings, self.property_embeddings)
        similarities = util.cos_sim(key_embeddings, self.property_embeddings)

        for i, relation in enumerate(extracted_relations):
            best_idx = int(similarities[i].argmax())
            best_score = float(similarities[i][best_idx])
            if best_score < self.threshold:
                continue
            match = self.ontology.properties[best_idx]
            best_matches.append(RelationMatch(relation, match, best_score))

        return best_matches


def label_alias_embedding_rl(inputs: Dict[str, Data], outputs: Dict[str, Data], model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.5):

    ontology_path = os.environ.get("ONTOLOGY_PATH", "false")
    if ontology_path == "false":
        raise ValueError("ONTOLOGY_PATH is not set")
    else:
        ontology_path = ontology_path
        
    linker = AliasAndTransformerBasedRelationLinker(ontology_path, model_name=model_name, threshold=threshold)
    
    if os.path.isdir(inputs["source"].path):
        os.makedirs(outputs["output"].path, exist_ok=True)
        for file in tqdm(os.listdir(inputs["source"].path), desc="Linking relations"):
            te_doc_in  = TE_Document(**json.load(open(os.path.join(inputs["source"].path, file))))
            relation_texts = list({triple.predicate.surface_form for triple in te_doc_in.triples if triple.predicate.surface_form})
            relation_matches = linker.link_relations(relation_texts)
            te_links = [TE_Pair(span=match.relation, mapping=match.predicate.uri, link_type="predicate", score=match.score) for match in relation_matches]
            te_doc_out = te_doc_in.model_copy(deep=True)
            te_doc_out.links += te_links
            with open(os.path.join(outputs["output"].path, file), "w") as f:
                f.write(te_doc_out.model_dump_json())
    else:
        te_doc_in  = TE_Document(**json.load(open(inputs["source"].path))) # TODO: check if this is correct
        relation_matches = linker.link_relations(list({triple.predicate.surface_form for triple in te_doc_in.triples if triple.predicate.surface_form}))
        te_links = [TE_Pair(span=match.relation, mapping=match.predicate.uri, link_type="predicate", score=match.score) for match in relation_matches]
        te_doc_out = te_doc_in.model_copy(deep=True)
        te_doc_out.links += te_links
        with open(outputs["output"].path, "w") as f:
            f.write(te_doc_out.model_dump_json())