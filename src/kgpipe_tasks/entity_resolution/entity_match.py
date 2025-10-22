from kgpipe.common import KG, DataFormat, KgTask, Data, Registry
from rdflib import Graph, URIRef, Literal, RDFS, OWL
from sentence_transformers import SentenceTransformer
from typing import List
from sentence_transformers import util
from pathlib import Path
from typing import Dict
import os
from tqdm import tqdm
import json
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document, TE_Pair

from kgpipe.util.embeddings import global_encode



def normalize(text):
    return text.replace('_', ' ').replace('-', ' ').strip().lower()


class EntityMatch:
    def __init__(self, surfaceform: str, label_uri: str, score: float):
        self.surfaceform = surfaceform
        self.label_uri = label_uri
        self.score = score

class LabelBasedEntityLinker:

    def __init__(self, kg_path: Path):
        self.kg = kg_path

        graph = Graph().parse(kg_path)

        self.entity_with_labels = []

        LABEL_URI = URIRef("http://www.w3.org/2000/01/rdf-schema#label")

        for s, _, o in graph.triples((None, LABEL_URI, None)):
            self.entity_with_labels.append((str(s), str(o)))

        entity_labels = [label for _, label in self.entity_with_labels]
        self.label_embeddings = global_encode(entity_labels)
        # print(self.label_embeddings.shape)
        self.match_cache = {}

    def link_entities(self, enitity_surfaceform: List[str]) -> List[EntityMatch]:
        if len(enitity_surfaceform) == 0:
            return []

        surfaceform_texts = [normalize(k) for k in enitity_surfaceform]
        surfaceform_embeddings = global_encode(surfaceform_texts)
        # surfaceform_embeddings = surfaceform_embeddings.to(self.label_embeddings.device)

        if surfaceform_embeddings.shape[1] != self.label_embeddings.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: {surfaceform_embeddings.shape[1]} vs {self.label_embeddings.shape[1]}")

        # 6. Match keys to ontology
        similarities = util.cos_sim(surfaceform_embeddings, self.label_embeddings)


        best_matches = []

        # 7. Display best match for each key
        for i, key in enumerate(enitity_surfaceform):
            best_idx = int(similarities[i].argmax())
            # INSERT_YOUR_CODE
            # Print the 5 best matches for this key
            # top5_idx = similarities[i].topk(5).indices.tolist()
            # print(f"\nTop 5 matches for '{key}':")
            # for rank, idx in enumerate(top5_idx):
            #     entity_uri = self.entity_with_labels[idx][0]
            #     entity_label = self.entity_with_labels[idx][1]
            #     score = float(similarities[i][idx])
            #     print(f"  {rank+1}. {entity_label} ({entity_uri}) - score: {score:.4f}")

            best_score = float(similarities[i][best_idx])
            match = self.entity_with_labels[best_idx][0]
            
            # print(f"\nProperty Key: '{key}'")
            # print(f"Best Match:   {match.uri}")
            # print(f"Label:        {match.label}")
            # print(f"Score:        {best_score:.4f}")
        
            best_matches.append(EntityMatch(key, match, best_score))


        return best_matches


@Registry.task(
    input_spec={"kg": DataFormat.RDF_NTRIPLES, "er_document": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Link entities to their labels",
    category=["EntityResolution", "EntityMatching"]
)
def label_based_entity_linker(inputs: Dict[str, Data], outputs: Dict[str, Data]):

    linker = LabelBasedEntityLinker(inputs["kg"].path)
    
    def link_entities(er_document: TE_Document):
        entity_surfaceforms = list(set([triple.subject.surface_form for triple in er_document.triples] + [triple.object.surface_form for triple in er_document.triples]))
        
        # print(entity_surfaceforms)
        matches = linker.link_entities(entity_surfaceforms)
        te_links = [TE_Pair(span=match.surfaceform, mapping=match.label_uri, link_type="entity", score=match.score) for match in matches]
        return te_links


    if os.path.isdir(inputs["er_document"].path):
        os.makedirs(outputs["output"].path, exist_ok=True)
        for file in tqdm(os.listdir(inputs["er_document"].path), desc="Linking relations"):
            te_doc_in  = TE_Document(**json.load(open(os.path.join(inputs["er_document"].path, file))))
            te_links = link_entities(te_doc_in)
            te_doc_in = TE_Document(links=te_links)
            with open(os.path.join(outputs["output"].path, file), "w") as f:
                f.write(te_doc_in.model_dump_json())
    else:
        te_doc_in  = TE_Document(**json.load(open(inputs["er_document"].path)))
        te_links = link_entities(te_doc_in)
        te_doc_in = TE_Document(links=te_links)
        with open(outputs["output"].path, "w") as f:
            f.write(te_doc_in.model_dump_json())




if __name__ == "__main__":
    kg = KG(id="0", name="seed", path=Path("/home/marvin/project/data/films_1k/sources/seed.nt"), format=DataFormat.RDF_NTRIPLES)
    linker = LabelBasedEntityLinker(kg)
    entities = ["Mary Anderson", "Scooby Doo"]
    matches = linker.link_entities(entities)
    for match in matches:
        print(match.entity, match.label, match.score)   