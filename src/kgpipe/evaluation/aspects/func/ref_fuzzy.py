from kgpipe.common import KG
from pathlib import Path
from typing import Dict, Literal
from rdflib import Graph, URIRef, SKOS, RDFS
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
from kgpipe.util.embeddings.st_emb import get_model

THRESHOLD = 0.95


def load_kg_labels(ttl_path: str):
    g = Graph(); g.parse(ttl_path)
    items = []  # (uri, text)
    for s, _, o in g.triples((None, SKOS.prefLabel, None)):
        items.append((str(s), str(o)))
    for s, _, o in g.triples((None, SKOS.altLabel, None)):
        items.append((str(s), str(o)))
    for s, _, o in g.triples((None, RDFS.label, None)):
        items.append((str(s), str(o)))
    # de-duplicate by (uri,text)
    seen = set(); dedup = []
    for u,t in items:
        if (u,t) not in seen:
            seen.add((u,t)); dedup.append((u,t))
    return dedup

def normalize(vectors: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / n


# --- Index build ---
class LabelIndex:

    show_progress_bar = False

    def __init__(self, items, model):
        self.uris  = [u for u,_ in items]
        self.texts = [t for _,t in items]
        self.model = model
        self.vecs  = normalize(self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=self.show_progress_bar))

    def update_index(self, items: list[tuple[str, str]]):
        self.uris.extend([u for u,_ in items])
        self.texts.extend([t for _,t in items])
        self.vecs = normalize(self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=self.show_progress_bar))

    def search(self, query_text: str, k: int = 10):
        q = normalize(self.model.encode([query_text], convert_to_numpy=True, show_progress_bar=self.show_progress_bar))[0]
        sims = self.vecs @ q  # cosine (since normalized)
        idx = np.argsort(-sims)[:k]
        return [(self.uris[i], self.texts[i], float(sims[i])) for i in idx]

    def search_all(self, query_text: list[str], k: int = 10):
        q = normalize(self.model.encode(query_text, convert_to_numpy=True, show_progress_bar=self.show_progress_bar))
        sims = self.vecs @ q  # cosine (since normalized)
        idx = np.argsort(-sims, axis=1)[:, :k]
        return [(self.uris[i], self.texts[i], float(sims[i][j])) for i, j in enumerate(idx)]


class LinkDecision(BaseModel):
    uri: str
    label: str
    score: float
    decision: Literal["accept", "review"]

class SimpleGraphLinker:
    def __init__(self, ttl_path: str, model = None):
        self.model = model or get_model()
        items = load_kg_labels(ttl_path)
        self.index = LabelIndex(items, self.model)
        self.THRESHOLD = THRESHOLD

    def update_index(self, new_aliases: list[tuple[str, str]]):
        if new_aliases:
            self.index.update_index(new_aliases)


    def link(self, graph: Graph):

        linked_entities: Dict[str, LinkDecision] = {}

        for s, _, o in graph.triples((None, RDFS.label, None)):
            uri = str(s)
            label = str(o)

            candidates = self.index.search(label, k=5)
            best_uri, best_text, best_score = candidates[0]
            decision = "accept" if best_score >= self.THRESHOLD else "review"

            if not uri in linked_entities:
                entry = LinkDecision(
                    uri=best_uri,
                    label=best_text,
                    score=best_score,
                    decision=decision
                )
                linked_entities[uri] = entry
            else:
                current_score = linked_entities[uri].score
                if best_score > current_score:
                    linked_entities[uri] = LinkDecision(
                        uri=best_uri,
                        label=best_text,
                        score=best_score,
                        decision=decision
                    )

        return linked_entities


def _match_entities_with_label_embeddings(test_kg: KG, reference_kg_path: Path) -> Dict[str, LinkDecision]:

    test_graph = Graph()
    test_graph.parse(test_kg.path, format="nt")
    reference_graph = Graph()
    reference_graph.parse(reference_kg_path, format="nt")

    linker = SimpleGraphLinker(reference_kg_path.as_posix(), get_model())
    linked_entities = linker.link(test_graph)
    return linked_entities


def match_entities_with_paris_algorithm(test_kg: KG, reference_kg_path: Path) -> Dict[str, LinkDecision]:
    raise NotImplementedError("Not implemented")


def match_entities(test_kg: KG, reference_kg_path: Path, method: Literal["label_embeddings", "paris"] = "label_embeddings") -> Dict[str, LinkDecision]:
    if method == "label_embeddings":
        return _match_entities_with_label_embeddings(test_kg, reference_kg_path)
    elif method == "paris":
        return match_entities_with_paris_algorithm(test_kg, reference_kg_path)


def map_entities(matched_entities: Dict[str, LinkDecision], test_kg: Graph) -> Graph:

    mapped_graph = Graph()
    
    for s, p, o in test_kg:
        s_str = str(s)
        o_str = str(o)
        if s_str in matched_entities:
            s_str = matched_entities[s_str].uri
        if isinstance(o, URIRef) and o_str in matched_entities:
            o_str = matched_entities[o_str].uri
            mapped_graph.add((URIRef(s_str), p, URIRef(o_str)))
        else:
            mapped_graph.add((URIRef(s_str), p, o))

    return mapped_graph


if __name__ == "__main__":
    from kgpipe.common.models import DataFormat
    test_kg = KG(id="test_kg", name="test_kg", path=Path("/home/marvin/project/data/old/acquisiton/film1k_bundle/split_0/kg/seed/data.nt"), format=DataFormat.RDF_NTRIPLES)
    reference_kg_path = Path("/home/marvin/project/data/old/acquisiton/film1k_bundle/split_0/kg/seed/data.nt")
    matched_entities = match_entities(test_kg, reference_kg_path)
    mapped_graph = map_entities(matched_entities, test_kg.get_graph())
    mapped_graph.serialize(destination="mapped_graph.nt", format="nt")