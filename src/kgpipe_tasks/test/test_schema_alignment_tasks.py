import os.path

from kgpipe.common import Data, DataFormat
from kgpipe_tasks.schema_alignment.schema_matching.valentine_schema_matching import valentine_csv_matching
from kgpipe_tasks.schema_alignment.ontology_matching.agreementmaker_matching import agreementmaker_ontology_matching
from . import get_test_data_path

import pytest
import tempfile
import json

@pytest.mark.docker
def test_valentine_schema_matching():
    source_csv_path = get_test_data_path("csv/source.csv")
    target_csv_path = get_test_data_path("csv/target.csv")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".er.json")

    data_source_csv = Data(source_csv_path, DataFormat.CSV)
    data_target_csv = Data(target_csv_path, DataFormat.CSV)
    data_output = Data(output_file.name, DataFormat.ER_JSON)

    report = valentine_csv_matching.run(
        [data_source_csv, data_target_csv],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    #TODO
    expected_content = {
        "matches": [
            {
                "id_1": "author",
                "id_2": "author",
                "score": 0.6,
                "id_type": "str"
            },
            {
                "id_1": "title",
                "id_2": "title",
                "score": 0.14285714285714285,
                "id_type": "str"
            }
        ],
        "blocks": [],
        "clusters": []
    }

    actual_content = ""
    with open(output_file.name, 'r', encoding='utf-8') as f:
        actual_content = json.load(f)

    assert actual_content == expected_content

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

@pytest.mark.skip(reason="Missing Task Definition")
def test_extract_ontology_from_rdf():
    pass

@pytest.mark.docker
def test_agreementmaker_ontology_matching():

    source_rdf_path = get_test_data_path("rdf/source.rdf")
    target_rdf_path = get_test_data_path("rdf/target.rdf")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".rdf")

    data_source_rdf = Data(source_rdf_path, DataFormat.RDF)
    data_target_rdf = Data(target_rdf_path, DataFormat.RDF)
    data_output = Data(output_file.name, DataFormat.AGREEMENTMAKER_RDF)

    report = agreementmaker_ontology_matching.run(
        [data_source_rdf, data_target_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 804

    print(json.dumps(report.__dict__, indent=4, default=str))

@pytest.mark.skip(reason="Missing Task Definition")
def test_limes_rdf_matching():
    pass