import os.path

import pytest
import tempfile
import json

from kgpipe.common import Data, DataFormat
from kgpipe_tasks.construction.construct import construct_rdf_from_te_json
from kgpipe_tasks.construction.json_to_rdf import construct_rdf_from_json2
from kgpipe_tasks.construction.json_processing import construct_rdf_from_json, construct_te_document_from_json
from kgpipe_tasks.construction.mapping import map_jsonpaths_to_rdf
from . import get_test_data_path

@pytest.mark.skip(reason="Not implemented yet")
def test_construct_rdf_from_te_json():

    te_json_path = get_test_data_path("json/te.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_te_json = Data(te_json_path, DataFormat.TE_JSON)
    data_output = Data(output_file.name, DataFormat.RDF_NTRIPLES)

    report = construct_rdf_from_te_json.run(
        [data_te_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

@pytest.mark.skip(reason="Not implemented yet")
def test_construct_rdf_from_json():
    json_path = get_test_data_path("json/json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_json = Data(json_path, DataFormat.JSON)
    data_output = Data(output_file.name, DataFormat.RDF_NTRIPLES)

    report = construct_rdf_from_json.run(
        [data_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_construct_te_document_from_json():
    json_path = get_test_data_path("json/dbp_json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".te.json")

    data_json = Data(json_path, DataFormat.JSON)
    data_output = Data(output_file.name, DataFormat.TE_JSON)

    report = construct_te_document_from_json.run(
        [data_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"
    assert os.path.getsize(output_file.name) == 4096

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

@pytest.mark.skip(reason="Not implemented yet")
def test_map_jsonpaths_to_rdf():

    json_path = get_test_data_path("json/json")
    mapping_json_path = get_test_data_path("json/mapping.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_json = Data(json_path, DataFormat.JSON)
    data_mapping_json = Data(mapping_json_path, DataFormat.JSON)
    data_output = Data(output_file.name, DataFormat.TE_JSON)

    report = map_jsonpaths_to_rdf.run(
        [data_json, data_mapping_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir


def test_construct_rdf_from_json2():
    json_path = get_test_data_path("json/dbp_json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_json = Data(json_path, DataFormat.JSON)
    data_output = Data(output_file.name, DataFormat.RDF_NTRIPLES)

    report = construct_rdf_from_json2.run(
        [data_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"
    assert os.path.getsize(output_file.name) == 3964

    with open(os.path.join(json_path, "dbp-movie_depth=1.json"), encoding="utf-8") as f:
        input_json = json.load(f)

    with open(output_file.name, encoding="utf-8") as f:
        content = f.read()
        print(content)

    assert len(content.strip().splitlines()) == 32

    assert f"\"{input_json['runtime']}\"" in content

    assert f"\"{input_json['producer']}\"" in content

    for actor in input_json["starring"]:
        assert f"\"{actor.replace("\\", "\\\\").replace('"', '\\"')}\"" in content

    for name in input_json["name"]:
        assert f"\"{name}\"" in content

    assert f"\"{input_json['film_title']}\"" in content

    assert f"\"{input_json['director']['name']}\"" in content
    assert f"\"{input_json['director']['birth_date']}\"" in content
    assert f"\"{input_json['director']['death_date']}\"" in content

    assert f"\"{input_json['music']}\"" in content

    assert f"\"{input_json['distributor']}\"" in content

    print(json.dumps(report.__dict__, indent=4, default=str))
    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir
