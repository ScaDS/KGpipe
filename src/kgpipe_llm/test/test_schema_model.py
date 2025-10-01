from kgpipe_llm.construct.schema_models import JsonLDDocument, JSON_LD_SCHEMA_DICT

from rdflib import Graph
import json

def test_schema_model():
    kg_path = "/home/marvin/project/data/acquisiton/film1k_bundle/split_0/kg/seed/data_h10.nt"

    graph = Graph()
    graph.bind("schema", "http://schema.org/")
    graph.bind("kgo", "https://kg.org/ontology/")
    graph.parse(kg_path, format="nt")

    json_ld = graph.serialize(format="json-ld")
    json_data = json.loads(json_ld)
    
    if isinstance(json_data, list):
        for item in json_data:
            jsonldoc = JsonLDDocument(**item)
            print(jsonldoc)
    else:
        jsonldoc = JsonLDDocument(**json_data)
        print(jsonldoc)


    
    # jsonldoc = JsonLDDocument(**json_data)
    # print(jsonldoc)
    