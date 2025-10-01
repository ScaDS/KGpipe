import os
import json
from sentence_transformers import SentenceTransformer, util
from rdflib import Graph, OWL, RDFS, RDF
from typing import Dict, List
from kgpipe.common.models import KgTask, Data, DataFormat
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document, TE_Pair
# from kgflex.resources.mainspec import *
# from kgflex.framework.kgflex import *
# import json
import logging

logger = logging.getLogger(__name__)

# def spacy_relation_linking():

#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     extracted_predicates = [] # ["bornOn", "bornIn", "developed"]

#     with open("/data/results/foundation-eval/temporary/Angela_Merkel.txt.tsv", "r") as f:
#         for line in f:
#             extracted_predicates.append(line.strip().split("\t")[1])

#     extracted_predicates = list(set(extracted_predicates))

#     ontology_predicates_dbpedia = {}
#     with open("/data/results/foundation-eval/temporary/ontology_props.tsv", "r") as f:
#         for line in f:
#             uri, predicate = line.strip().split("\t")
#             ontology_predicates_dbpedia[predicate] = uri

#     # Compute embeddings for extracted predicates
#     extracted_embeddings = model.encode(extracted_predicates, convert_to_tensor=True)

#     # Compute embeddings for ontology predicates
#     ontology_labels = list(ontology_predicates_dbpedia.keys())
#     ontology_embeddings = model.encode(ontology_labels, convert_to_tensor=True)

#     # Compute cosine similarity
#     cosine_scores = util.cos_sim(extracted_embeddings, ontology_embeddings)

#     results = []

#     # Print mapping results
#     for idx, extracted_pred in enumerate(extracted_predicates):
#         scores = cosine_scores[idx]
#         best_match_idx = scores.argmax().item()
#         best_match_label = ontology_labels[best_match_idx]
#         best_match_uri = ontology_predicates_dbpedia[best_match_label]
        
#         print(f"'{extracted_pred}' matched with '{best_match_label}' ({best_match_uri}) "
#             f"with similarity: {scores[best_match_idx]:.4f}")

#         results.append({
#             "surfaceForm": extracted_pred,
#             "label": best_match_label,
#             "uri": best_match_uri,
#             "similarity": scores[best_match_idx].item()
#         })
    
#     for r in results:
#         print(r)

# def test_spacy_relation_linking():
#     spacy_relation_linking()


# def link_relations_in_te_json(TE_JSON):
#     with open(TE_JSON, "r") as f:
#         data = json.load(f)
#         te_doc = TE_Document(**data)

#     extracted_relations = [ triple.predicate.surface_form for triple in te_doc.triples ]

class SimpleTransformerBasedRelationLinker:

    def __init__(self, ontology_file, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        g = Graph()
        g.parse(ontology_file)

        ontology_relations = {}
        # Extract ObjectProperties
        for s in g.subjects(RDF.type, OWL.ObjectProperty):
            label = g.value(s, RDFS.label)
            if label:
                ontology_relations[str(label)] = str(s)

        # Extract DatatypeProperties
        for s in g.subjects(RDF.type, OWL.DatatypeProperty):
            label = g.value(s, RDFS.label)
            if label:
                ontology_relations[str(label)] = str(s)

        self.ontology_relations = ontology_relations

        # Compute embeddings for ontology predicates
        self.ontology_labels = list(ontology_relations.keys())
        self.ontology_embeddings = self.model.encode(self.ontology_labels, convert_to_tensor=True)

    def link_relations(self, extracted_relations) -> List[TE_Pair]:
        if len(extracted_relations) == 0:
            logger.error("No extracted relations")
            return []
        
        extracted_embeddings = self.model.encode(extracted_relations, convert_to_tensor=True, show_progress_bar=False)
        # Ensure both tensors are on the same device
        self.ontology_embeddings = self.ontology_embeddings.to(extracted_embeddings.device)
        
        # Ensure both tensors have the same embedding dimension
        if extracted_embeddings.shape[1] != self.ontology_embeddings.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: {extracted_embeddings.shape[1]} vs {self.ontology_embeddings.shape[1]}")

        cosine_scores = util.cos_sim(extracted_embeddings, self.ontology_embeddings)


        links = []
        # Print mapping results
        for idx, extracted_pred in enumerate(extracted_relations):
            scores = cosine_scores[idx]
            best_match_idx = int(scores.argmax().item())
            best_match_label = self.ontology_labels[best_match_idx]
            best_match_uri = self.ontology_relations[best_match_label]

            links.append(TE_Pair(span=extracted_pred, mapping=best_match_uri, link_type="predicate", score=scores[best_match_idx].item()))

        return links




def transrel(inputs: Dict[str, Data], outputs: Dict[str, Data]):

    dir_or_file = inputs["source"].path

    simpleLinker = SimpleTransformerBasedRelationLinker("/home/marvin/project/data/current/ontology.ttl")

    if os.path.isdir(dir_or_file):
        os.makedirs(outputs["output"].path, exist_ok=True)
        for file in os.listdir(dir_or_file):
            te_doc_in  = TE_Document(**json.load(open(os.path.join(dir_or_file, file))))

            links = simpleLinker.link_relations(list({triple.predicate.surface_form for triple in te_doc_in.triples}))

            te_doc_out = TE_Document(links=links)
            with open(os.path.join(outputs["output"].path, file), "w") as f:
                f.write(te_doc_out.model_dump_json())
    else:
        te_doc_in  = TE_Document(**json.load(open(dir_or_file)))

        links = simpleLinker.link_relations([ triple.predicate.surface_form for triple in te_doc_in.triples ])

        te_doc_out = TE_Document(links=links)
        with open(outputs["output"].path, "w") as f:
            f.write(te_doc_out.model_dump_json())
        
    logger.info(f"Converted {dir_or_file} to {outputs['output'].path}")


transrel_task = KgTask(
    name="transformer_relation_linking",
    function=transrel,
    input_spec={"source": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.TE_JSON},
)

# if __name__ == "__main__":
#     pass
#     # links = link_relations("/data/abstracts_Person.100.openie.tejson/Dennis_Peacock.nt.json","/home/marvin/papers/odibel/film_spec.ttl")
#     # te_doc = TE_Document(links=links)

#     # with open("/data/abstracts_Person.100.customrl.tejson/Dennis_Peacock.nt.json", "w") as f:
#     #     f.write(te_doc.model_dump_json())

