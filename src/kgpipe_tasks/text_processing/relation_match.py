from sentence_transformers import util
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from typing import List, Dict, Tuple

import os
import json
from rdflib import Graph, RDF, RDFS, OWL
from collections import defaultdict
from tqdm import tqdm
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Pair, TE_Document
from kgpipe.common import Data, DataFormat, Registry
from kgcore.model.ontology import Ontology, OwlProperty, OntologyUtil

from kgpipe.util.embeddings import global_encode



# A simple fuzzy relation match

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
    Uses a transformer model to embed extracted relations and ontology predicates.
    Each predicate includes multiple aliases and one label.
    For matching, it computes cosine similarity between the extracted relation and
    each alias/label separately, then returns the predicate with the highest similarity.
    """


        
    def __init__(self, ontology_file, model_name="all-MiniLM-L6-v2"):
        self.ontology = OntologyUtil.load_ontology_from_file(ontology_file)
        # self.model = SentenceTransformer(model_name)

        property_texts = [build_property_text(p) for p in self.ontology.properties]
        print("number of properties: ", len(property_texts))
        self.property_embeddings = global_encode(property_texts)
        self.match_cache = {}

    def link_relations(self, extracted_relations: List[str]) -> List[RelationMatch]:
        if len(extracted_relations) == 0:
            # print("No extracted relations")
            return []

        # if extracted_relations in self.match_cache:
        #     return self.match_cache[extracted_relations]

        best_matches = []

        property_keys = extracted_relations
        key_texts = [normalize(k) for k in property_keys]
        key_embeddings = global_encode(key_texts)
        # key_embeddings = key_embeddings.to(self.property_embeddings.device)


        if key_embeddings.shape[1] != self.property_embeddings.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: {key_embeddings.shape[1]} vs {self.property_embeddings.shape[1]}")

        # 6. Match keys to ontology
        similarities = util.cos_sim(key_embeddings, self.property_embeddings)

        # 7. Display best match for each key
        for i, key in enumerate(property_keys):
            best_idx = int(similarities[i].argmax())
            best_score = float(similarities[i][best_idx])
            match = self.ontology.properties[best_idx]
            
            # print(f"\nProperty Key: '{key}'")
            # print(f"Best Match:   {match.uri}")
            # print(f"Label:        {match.label}")
            # print(f"Score:        {best_score:.4f}")
            best_matches.append(RelationMatch(key, match, best_score))

        return best_matches


@Registry.task(
    input_spec={"source": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Link relations using transformer embeddings on property label+alias ",
    category=["TextProcessing", "RelationLinking"]
)
def label_alias_embedding_rl(inputs: Dict[str, Data], outputs: Dict[str, Data]):

    ontology_path = os.environ.get("ONTOLOGY_PATH", "false")
    if ontology_path == "false":
        raise ValueError("ONTOLOGY_PATH is not set")
    else:
        ontology_path = ontology_path
        
    linker = AliasAndTransformerBasedRelationLinker(ontology_path)
    
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


# def test_relation_linking():
#     linker = AliasAndTransformerBasedRelationLinker("/home/marvin/project/data/current/ontology.ttl")

#     extracted_relations = ["starring", "music by", "directed by", "film genre"]
#     expected_matches = ["cast member", "composer", "director", "genre"]

#     result = linker.link_relations(extracted_relations)

#     actual_matches = [prop.label for prop in result]

#     [print(prop.uri + " " + prop.label+" "+prop.type.value) for prop in result]

#     assert actual_matches == expected_matches, f"Expected {expected_matches}, but got {result}"


# class SimpleTransformerBasedRelationLinker:

#     def __init__(self, ontology_file, model_name="all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)

#         g = Graph()
#         g.parse(ontology_file)

#         ontology_relations = {}
#         # Extract ObjectProperties
#         for s in g.subjects(RDF.type, OWL.ObjectProperty):
#             label = g.value(s, RDFS.label)
#             if label:
#                 ontology_relations[str(label)] = str(s)

#         # Extract DatatypeProperties
#         for s in g.subjects(RDF.type, OWL.DatatypeProperty):
#             label = g.value(s, RDFS.label)
#             if label:
#                 ontology_relations[str(label)] = str(s)

#         self.ontology_relations = ontology_relations

#         # Compute embeddings for ontology predicates
#         self.ontology_labels = list(ontology_relations.keys())
#         self.ontology_embeddings = self.model.encode(self.ontology_labels, convert_to_tensor=True)

#     def link_relations(self, extracted_relations) -> List[TE_Pair]:
#         if len(extracted_relations) == 0:
#             print("No extracted relations")
#             return []
        
#         extracted_embeddings = self.model.encode(extracted_relations, convert_to_tensor=True, show_progress_bar=False)
#         # Ensure both tensors are on the same device
#         self.ontology_embeddings = self.ontology_embeddings.to(extracted_embeddings.device)
        
#         # Ensure both tensors have the same embedding dimension
#         if extracted_embeddings.shape[1] != self.ontology_embeddings.shape[1]:
#             raise ValueError(f"Embedding dimension mismatch: {extracted_embeddings.shape[1]} vs {self.ontology_embeddings.shape[1]}")

#         cosine_scores = util.cos_sim(extracted_embeddings, self.ontology_embeddings)


#         links = []
#         # Print mapping results
#         for idx, extracted_pred in enumerate(extracted_relations):
#             scores = cosine_scores[idx]
#             best_match_idx = int(scores.argmax().item())
#             best_match_label = self.ontology_labels[best_match_idx]
#             best_match_uri = self.ontology_relations[best_match_label]

#             links.append(TE_Pair(span=extracted_pred, mapping=best_match_uri, link_type="predicate", score=scores[best_match_idx].item()))

#         return links




# def transrel(inputs: Dict[str, Data], outputs: Dict[str, Data]):

#     dir_or_file = inputs["source"].path

#     simpleLinker = SimpleTransformerBasedRelationLinker("/home/marvin/project/data/current/ontology.ttl")

#     if os.path.isdir(dir_or_file):
#         os.makedirs(outputs["output"].path, exist_ok=True)
#         for file in os.listdir(dir_or_file):
#             te_doc_in  = TE_Document(**json.load(open(os.path.join(dir_or_file, file))))

#             links = simpleLinker.link_relations(list({triple.predicate.surface_form for triple in te_doc_in.triples}))

#             te_doc_out = TE_Document(links=links)
#             with open(os.path.join(outputs["output"].path, file), "w") as f:
#                 f.write(te_doc_out.model_dump_json())
#     else:
#         te_doc_in  = TE_Document(**json.load(open(dir_or_file)))

#         links = simpleLinker.link_relations([ triple.predicate.surface_form for triple in te_doc_in.triples ])

#         te_doc_out = TE_Document(links=links)
#         with open(outputs["output"].path, "w") as f:
#             f.write(te_doc_out.model_dump_json())
        
#     print(f"Converted {dir_or_file} to {outputs['output'].path}")



# Later a refinement step based on entity type information

from pydantic import BaseModel, create_model
from typing import Any, Dict, List, get_origin, get_args

def dict_to_pydantic_model(name: str, data: Dict[str, Any]) -> BaseModel:
    """Generate a Pydantic model from a dict with inferred field types."""
    
    def infer_type(value):
        if isinstance(value, dict):
            return dict  # fallback for nested dict
        elif isinstance(value, list):
            if value:
                return List[type(value[0])]
            return List[Any]
        else:
            return type(value)

    fields = {k: (infer_type(v), ...) for k, v in data.items()}
    return create_model(name, **fields)

def dict_to_json_schema(data: Dict[str, Any], name: str = "AutoModel") -> Dict[str, Any]:
    model = dict_to_pydantic_model(name, data)
    return model.model_json_schema()

if __name__ == "__main__":

    # path_to_json = "/home/marvin/project/data/acquisiton/film1k_bundle/split_0/sources/json/data/0047b37c2629a73234fe90c729348b8a.json"

    json_dict = {
        "name": "John Doe",
        "age": 30,
        "is_student": True,
        "courses": ["Math", "Science"],
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "zip": "12345"
        }
    }

    schema = dict_to_json_schema(json_dict)
    print(json.dumps(schema, indent=4))

