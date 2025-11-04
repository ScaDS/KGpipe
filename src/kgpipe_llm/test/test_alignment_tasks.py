from kgpipe_llm.alignment.json_alignment import map_json_to_ontology, apply_jsonpath_mapping
from kgcore.model.ontology import Ontology, OntologyUtil
import pytest
from pathlib import Path
from kgpipe.common import Data, DataFormat
from kgpipe_llm.alignment.json_alignment import align_kg
import tempfile

@pytest.mark.skip(reason="TODO")
def test_json_alignment():
    json_data = {
        "id": "tt0468569",
        "title": "The Dark Knight",
        "directors": [ {"name": "Christopher Nolan", "birthYear": 1970} ],
        "year": 2008
    }
    ontology = OntologyUtil.load_ontology_from_file(Path("/home/marvin/project/data/acquisiton/film1k_bundle/ontology.ttl"))
    result = map_json_to_ontology(json_data, ontology)
    if result is not None:
        print(result.model_dump_json(indent=4)) 
        apply_jsonpath_mapping(result, json_data)
    else:
        print("No mapping found")

@pytest.mark.skip(reason="TODO")
def test_ontology_alignment():
    kg1_path = Path("/home/marvin/project/data/acquisiton/film1k_bundle/split_0/kg/seed/data_h10.nt")
    kg2_path = Path("/home/marvin/project/data/acquisiton/film1k_bundle/split_0/kg/seed/data_h10.nt")
    kg1_data = Data(kg1_path, DataFormat.RDF_NTRIPLES)
    kg2_data = Data(kg2_path, DataFormat.RDF_NTRIPLES)

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")
    output_path.close()
    output_data = Data(output_path.name, DataFormat.JSON)

    response = align_kg.run([kg1_data, kg2_data], [output_data], stable_files_override=True)

    print(open(output_path.name, "r").read())

