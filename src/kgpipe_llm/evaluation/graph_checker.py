
from kgpipe_llm.common.core import get_client_from_env
from rdflib import Graph, RDFS, Literal, URIRef
from typing import Tuple
from pydantic import BaseModel
from tqdm import tqdm
import time

class TripleCheckerResponse(BaseModel):
    is_correct: bool
    # reason: str

def getLabelByUri(g: Graph, lang: str = "en"):
    labelByUri = {}
    for s, p, o in g.triples((None, RDFS.label, None)):
        if isinstance(o, Literal):
            if o.language is None or o.language == lang:
                labelByUri[str(s)] = str(o.value)
    return labelByUri

def generate_surface_triple(g: Graph):
    labelByUri = getLabelByUri(g)
    surfaceTriples = []
    for s, p, o in g.triples((None, None, None)):
        s_surface = str(s)
        p_surface = str(p)
        o_surface = str(o)
        
        if s_surface in labelByUri:
            s_surface = labelByUri[s_surface]
        if p_surface in labelByUri:
            p_surface = labelByUri[p_surface]
        if o_surface in labelByUri:
            o_surface = labelByUri[o_surface]
        
        surfaceTriples.append((s_surface, p_surface, o_surface))
    return surfaceTriples

def check_triple(triple: Tuple[str, str, str]) -> TripleCheckerResponse:
    s, p, o = triple
    
    prompt = f"""
    You are a fact checker. You are given a triple and you need to check if it is correct.
    Return the response in the following format:
    {{
        "is_correct": bool,
    }}

    Fact: {s} {p} {o}
    """
    
    response = get_client_from_env().send_prompt(
        prompt=prompt,
        schema_class=TripleCheckerResponse,
    )
    return response

def check_graph(graph: Graph, s_filter = set()):
    surfaceTriples = generate_surface_triple(graph)
    checks = {}
    from multiprocessing import Pool

    filtered_surfaceTriples = []

    for triple in tqdm(surfaceTriples):
        if triple[0] in s_filter or len(s_filter) == 0:
            filtered_surfaceTriples.append(triple)

    with Pool(8) as p:
        results = list(tqdm(p.imap(check_triple, filtered_surfaceTriples), total=len(filtered_surfaceTriples)))
        for triple, response in zip(filtered_surfaceTriples, results):
            checks[triple] = response

    return checks


if __name__ == "__main__":
    graph = Graph()
    graph.parse("/home/marvin/project/data/final/film_100/split_0/kg/seed/data.nt", format="turtle")

    ontology_graph = Graph()
    ontology_graph.parse("/home/marvin/project/data/final/movie-ontology.ttl", format="turtle")

    

    for s, p, o in ontology_graph.triples((None, None, None)):
        graph.add((s, p, o))

    s_filter = set()
    s_filter.add("Fugitive Pieces (film)")

    start_time = time.time()

    checks = check_graph(graph)
    
    end_time = time.time()

    for check in checks:
        print(checks[check], check)

    print(f"Time taken: {end_time - start_time} seconds")
