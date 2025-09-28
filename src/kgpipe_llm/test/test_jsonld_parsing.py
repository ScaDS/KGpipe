from rdflib import Graph

def test_jsonld_parsing():
    path = "/home/marvin/project/data/llm_responses/20250823_091150_as_json_ld.json"

    graph = Graph()
    graph.parse(path, format="json-ld")

    print(graph.serialize(format="turtle"))
