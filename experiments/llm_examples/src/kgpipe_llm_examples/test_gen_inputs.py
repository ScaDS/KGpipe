import shutil
import hashlib
import os
from rdflib import Graph

BENCHMARK_DATA_DIR = "/home/marvin/project/data/final/film_1k"
INPUT_DATA_DIR = "/home/marvin/phd/kgpipe/experiments/llm_examples/inputs"

sampled_entities = [
    "http://dbpedia.org/resource/Kajraare",
    "http://dbpedia.org/resource/Prison_(1987_film)",
    "http://dbpedia.org/resource/Casper_Meets_Wendy",
    "http://dbpedia.org/resource/The_Nice_Guys",
    "http://dbpedia.org/resource/Paying_the_Price:_Killing_the_Children_of_Iraq",
    "http://dbpedia.org/resource/Benji_(1974_film)",
    "http://dbpedia.org/resource/The_Mummy's_Shroud",
    "http://dbpedia.org/resource/Violent_City",
    "http://dbpedia.org/resource/Toy_Story_4",
    "http://dbpedia.org/resource/Girlfriends_(1978_film)"
]

def hash_uri(uri: str) -> str:
    return hashlib.md5(uri.encode('utf-8')).hexdigest()

def generate_text_source():
    TEXT_SOURCE_DIR = os.path.join(BENCHMARK_DATA_DIR, "split_0/sources/text/data")

    for entity in sampled_entities:
        entity_hash = hash_uri(entity)
        entity_file = os.path.join(TEXT_SOURCE_DIR, f"{entity_hash}.txt")
        if not os.path.exists(entity_file):
            raise FileNotFoundError(f"Entity file {entity_file} not found")
        with open(entity_file, "r") as f:
            os.makedirs(os.path.join(INPUT_DATA_DIR, "text"), exist_ok=True)
            shutil.copy(entity_file, os.path.join(INPUT_DATA_DIR, "text", f"{entity_hash}.txt"))

def generate_rdf_subgraph(graph: Graph, entity: str) -> Graph:
    """
    Generate a subgraph for the given entity.
    Fetching all triples for the given entity, and transitive objects.
    """
    from rdflib import URIRef, BNode

    subgraph = Graph()
    visited = set()
    queue = [URIRef(entity)]

    while queue:
        current_entity = queue.pop(0)
        if current_entity in visited:
            continue
        visited.add(current_entity)

        # Keep triples where current entity is the subject.
        for triple in graph.triples((current_entity, None, None)):
            subgraph.add(triple)
            obj = triple[2]
            # Transitively expand through object resources (not literals).
            if isinstance(obj, (URIRef, BNode)) and obj not in visited:
                queue.append(obj)

        # # Include direct incoming relations for context.
        # for triple in graph.triples((None, None, current_entity)):
        #     subgraph.add(triple)

    return subgraph
    
def generate_rdf_source():
    RDF_SOURCE_FILE=os.path.join(BENCHMARK_DATA_DIR, "split_0/sources/rdf/data.nt")

    graph = Graph()
    graph.parse(RDF_SOURCE_FILE, format="nt")
    
    PREFIX = "http://kg.org/rdf/0/resource/"

    os.makedirs(os.path.join(INPUT_DATA_DIR, "rdf"), exist_ok=True)
    for entity in sampled_entities:
        subgraph = generate_rdf_subgraph(graph, PREFIX + hash_uri(entity))
        subgraph_file = os.path.join(INPUT_DATA_DIR, "rdf", f"{hash_uri(entity)}.nt")

        subgraph.serialize(subgraph_file, format="nt")

def generate_rdf_reference():
    RDF_SOURCE_FILE=os.path.join(BENCHMARK_DATA_DIR, "split_0/kg/reference/data.nt")

    graph = Graph()
    graph.parse(RDF_SOURCE_FILE, format="nt")
    
    PREFIX = "http://kg.org/resource/"

    os.makedirs(os.path.join(INPUT_DATA_DIR, "rdf_reference"), exist_ok=True)
    for entity in sampled_entities:
        subgraph = generate_rdf_subgraph(graph, PREFIX + hash_uri(entity))
        subgraph_file = os.path.join(INPUT_DATA_DIR, "rdf_reference", f"{hash_uri(entity)}.nt")

        subgraph.serialize(subgraph_file, format="nt")

def generate_json_source():
    JSON_SOURCE_DIR=os.path.join(BENCHMARK_DATA_DIR, "split_0/sources/json/data/")

    os.makedirs(os.path.join(INPUT_DATA_DIR, "json"), exist_ok=True)
    for entity in sampled_entities:
        entity_hash = hash_uri(entity)
        entity_file = os.path.join(JSON_SOURCE_DIR, f"{entity_hash}.json")
        if not os.path.exists(entity_file):
            raise FileNotFoundError(f"Entity file {entity_file} not found")
        shutil.copy(entity_file, os.path.join(INPUT_DATA_DIR, "json", f"{entity_hash}.json"))


def test_generate_input_data():
    generate_text_source()
    generate_rdf_source()
    generate_rdf_reference()
    generate_json_source()
