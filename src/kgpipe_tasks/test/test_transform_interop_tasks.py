import filecmp
import json
import os
import tempfile
import csv

from kgpipe.common import Data, DataFormat
from kgpipe_tasks.text_processing.text_extraction.corenlp_extraction import corenlp_exchange
from kgpipe_tasks.entity_resolution.matcher.paris_rdf_matcher import paris_exchange
from kgpipe_tasks.transform_interop.transform import transform_rdf_to_csv, transform2_rdf_to_csv
from kgpipe_tasks.transform_interop.aggregation import aggregate2_te_json, aggregate3_te_json
from kgpipe_tasks.text_processing.entity_linking.spotlight_entity_linking import dbpedia_spotlight_exchange
from kgpipe_tasks.text_processing.entity_linking.falcon_entity_linking import falcon_exchange
from . import get_test_data_path
import pytest

expected_header_source = {
    "subject",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type_uri",
    "http://example.org/bookstore/bookAuthor_uri",
    "http://example.org/bookstore/bookTitle_literal",
    "http://www.w3.org/2000/01/rdf-schema#label_literal",
    "http://example.org/bookstore/isbn13_literal",
    "http://example.org/bookstore/birthDate_literal",
    "http://example.org/bookstore/birthPlace_literal",
}
expected_header_target = {
    "subject",
    "http://example.org/ontology/birthDate_literal",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type_uri",
    "http://www.w3.org/2000/01/rdf-schema#label_literal",
    "http://example.org/ontology/birthPlace_literal",
    "http://example.org/ontology/isbn_literal",
    "http://example.org/ontology/title_literal",
    "http://example.org/ontology/author_uri",
}


expected_row_source = {
    "subject": "http://example.org/bookstore/itemD",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type_uri": "http://example.org/bookstore/Book",
    "http://example.org/bookstore/bookAuthor_uri": "http://example.org/bookstore/authorLee",
    "http://example.org/bookstore/bookTitle_literal": "To Kill a Mockingbird",
    "http://www.w3.org/2000/01/rdf-schema#label_literal": "itemD",
    "http://example.org/bookstore/isbn13_literal": "9780061120084",
    "http://example.org/bookstore/birthDate_literal": "",
    "http://example.org/bookstore/birthPlace_literal": "",
}

expected_row_target = {
    "subject": "http://example.org/library/authorOrwell",
    "http://example.org/ontology/birthDate_literal": "1903-06-25",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type_uri": "http://example.org/ontology/Author",
    "http://www.w3.org/2000/01/rdf-schema#label_literal": "George Orwell",
    "http://example.org/ontology/birthPlace_literal": "Motihari, Bengal Presidency, British India",
    "http://example.org/ontology/isbn_literal": "",
    "http://example.org/ontology/title_literal": "",
    "http://example.org/ontology/author_uri": "",
}


def test_transform2_rdf_to_csv():

    source1_nt_path = get_test_data_path("rdf/source.nt")
    source2_nt_path = get_test_data_path("rdf/target.nt")

    output_file1 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    output_file2 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

    data_source1_nt = Data(source1_nt_path, DataFormat.RDF_NTRIPLES)
    data_source2_nt = Data(source2_nt_path, DataFormat.RDF_NTRIPLES)
    data_output1 = Data(output_file1.name, DataFormat.CSV)
    data_output2 = Data(output_file2.name, DataFormat.CSV)

    report = transform2_rdf_to_csv.run(
        [data_source1_nt, data_source2_nt],
        [data_output1, data_output2],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file1.name) == 1513

    with open(output_file1.name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    assert set(fieldnames) == expected_header_source, "Wrong Header in file 1."

    found = False
    with open(output_file1.name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(row[col] == val for col, val in expected_row_source.items()):
                found = True
                break

    assert found, "Expected Line not found in file 2."

    assert os.path.getsize(output_file2.name) == 1488

    with open(output_file2.name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    assert set(fieldnames) == expected_header_target, "Wrong Header in file 2."

    found = False
    with open(output_file2.name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(row[col] == val for col, val in expected_row_target.items()):
                found = True
                break

    assert found, "Expected Line not found in file 2."
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir


def test_transform_rdf_to_csv():

    source_nt_path = get_test_data_path("rdf/source.nt")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

    data_source_nt = Data(source_nt_path, DataFormat.RDF_NTRIPLES)
    data_output = Data(output_file.name, DataFormat.CSV)

    report = transform_rdf_to_csv.run(
        [data_source_nt],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 1513

    with open(output_file.name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    assert set(fieldnames) == expected_header_source, "Wrong Header."

    found = False
    with open(output_file.name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(row[col] == val for col, val in expected_row_source.items()):
                found = True
                break

    assert found, "Expected Line not found."

    print(json.dumps(report.__dict__, indent=4, default=str))


    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_paris_exchange():

    source_paris_path = get_test_data_path("csv/paris")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".er.json")

    data_source_paris = Data(source_paris_path, DataFormat.PARIS_CSV)
    data_output = Data(output_file.name, DataFormat.ER_JSON)

    report = paris_exchange.run(
        [data_source_paris],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 1672

    assert filecmp.cmp(output_file.name, get_test_data_path("json/er.json"), shallow=False), \
        f"Wrong file contents"
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_dbpedia_spotlight_exchange():

    source_json_path = get_test_data_path("json/spotlight.json")

    output_dir = tempfile.mkdtemp()

    data_source_paris = Data(source_json_path, DataFormat.SPOTLIGHT_JSON)
    data_output = Data(output_dir, DataFormat.TE_JSON)

    report = dbpedia_spotlight_exchange.run(
        [data_source_paris],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
    assert len(files) == 1
    path = os.path.join(output_dir, files[0])
    assert os.path.getsize(path) == 2300

    with open(path, encoding="utf-8") as f:
        content = f.read()

    assert '{"span": "hobbit", "mapping": "http://dbpedia.org/resource/Hobbit", "score": 1.0, "link_type": "entity"}' in content

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_falcon_exchange():

    source_json_path = get_test_data_path("json/falcon.json")

    output_dir = tempfile.mkdtemp()

    data_source_paris = Data(source_json_path, DataFormat.FALCON_JSON)
    data_output = Data(output_dir, DataFormat.TE_JSON)

    report = falcon_exchange.run(
        [data_source_paris],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
    assert len(files) == 1
    path = os.path.join(output_dir, files[0])
    assert os.path.getsize(path) == 2910

    with open(path, encoding="utf-8") as f:
        content = f.read()

    assert '{"span": "The Hobbit", "mapping": "http://www.wikidata.org/entity/Q74229", "score": 1.0, "link_type": "entity"}' in content
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_aggregate2_te_json():

    #TODO - different te.json's
    source1_te_json_path = get_test_data_path("json/te.json")
    source2_te_json_path = get_test_data_path("json/te.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".te.json")

    data_source1_te_json = Data(source1_te_json_path, DataFormat.TE_JSON)
    data_source2_te_json = Data(source2_te_json_path, DataFormat.TE_JSON)
    data_output = Data(output_file.name, DataFormat.TE_JSON)

    report = aggregate2_te_json.run(
        [data_source1_te_json, data_source2_te_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 8915
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_aggregate3_te_json():

    #TODO - different te.json's
    source1_te_json_path = get_test_data_path("json/te.json")
    source2_te_json_path = get_test_data_path("json/te.json")
    source3_te_json_path = get_test_data_path("json/te.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".te.json")

    data_source1_te_json = Data(source1_te_json_path, DataFormat.TE_JSON)
    data_source2_te_json = Data(source2_te_json_path, DataFormat.TE_JSON)
    data_source3_te_json = Data(source3_te_json_path, DataFormat.TE_JSON)
    data_output = Data(output_file.name, DataFormat.TE_JSON)

    report = aggregate3_te_json.run(
        [data_source1_te_json, data_source2_te_json, data_source3_te_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 13349

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

@pytest.mark.skip(reason="Missing Task Definition")
def test_rdf_to_wide_csv():
    pass

def test_corenlp_exchange():
    openie_json_path = get_test_data_path("json/openie.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".te.json")

    data_json = Data(openie_json_path, DataFormat.OPENIE_JSON)
    data_output = Data(output_file.name, DataFormat.TE_JSON)

    report = corenlp_exchange.run(
        [data_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 2295
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir