from kgpipe.common import Data, DataFormat
from kgpipe_tasks.entity_resolution.matcher.paris_rdf_matcher import paris_entity_matching
from kgpipe_tasks.entity_resolution.matcher.jedai_tab_matcher import pyjedai_entity_matching
from kgpipe_tasks.entity_resolution.fusion.simple import fusion_union_rdf, union_matched_rdf, union_matched_rdf_combined, fusion_first_value, select_first_value
from kgpipe_tasks.entity_resolution.entity_match import label_based_entity_linker
from . import get_test_data_path

import pytest
import tempfile
import os
import json

# === PARIS ===
@pytest.mark.docker
def test_paris_entity_matching():
    source_rdf_path = get_test_data_path("rdf/source.nt")
    target_rdf_path = get_test_data_path("rdf/target.nt")

    output_dir = tempfile.mkdtemp()

    data_source_rdf = Data(source_rdf_path, DataFormat.RDF_NTRIPLES)
    data_target_rdf = Data(target_rdf_path, DataFormat.RDF_NTRIPLES)
    data_output = Data(output_dir, DataFormat.PARIS_CSV)

    report = paris_entity_matching.run(
        [data_source_rdf, data_target_rdf], 
        [data_output], 
        stable_files_override=True
    )

    assert report.status == "success"

    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]

    empty_files = [f for f in files if os.path.getsize(f) == 0]
    assert len(empty_files) == 67
    assert len(files) == 84
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

# === PYJEDAI ===
@pytest.mark.docker
def test_pyjedai_entity_matching():
    source_csv_path = get_test_data_path("csv/source.csv")
    target_csv_path = get_test_data_path("csv/target.csv")
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".er.json")

    data_source_csv = Data(source_csv_path, DataFormat.CSV)
    data_target_csv = Data(target_csv_path, DataFormat.CSV)
    data_output = Data(output_file.name, DataFormat.ER_JSON)

    report = pyjedai_entity_matching.run(
        [data_source_csv, data_target_csv],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    # TODO
    expected_content = {
    "matches": [
        {
            "id_1": "itemA",
            "id_2": "book1",
            "score": 0.468671603211667,
            "id_type": "entity"
        },
        {
            "id_1": "itemB",
            "id_2": "book2",
            "score": 0.868770796713726,
            "id_type": "entity"
        },
        {
            "id_1": "itemC",
            "id_2": "book3",
            "score": 0.45403544832634846,
            "id_type": "entity"
        }
    ],
    "blocks": [],
    "clusters": []
    }

    actual_content = ""
    with open(output_file.name, 'r', encoding='utf-8') as f:
        actual_content = json.load(f)

    assert actual_content == expected_content

    with open(output_file.name, 'r') as f:
        print(f.read())
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir


def test_fusion_union_rdf():
    source_nt_path = get_test_data_path("rdf/source.nt")
    target_nt_path = get_test_data_path("rdf/target.nt")
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_source_nt = Data(source_nt_path, DataFormat.RDF_NTRIPLES)
    data_target_nt = Data(target_nt_path, DataFormat.RDF_NTRIPLES)
    data_output = Data(output_file.name, DataFormat.RDF_NTRIPLES)

    report = fusion_union_rdf.run(
        [data_source_nt, data_target_nt],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 8203

    with open(output_file.name, 'r') as f:
        print(f.read())
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_union_matched_rdf():
    source_nt_path = get_test_data_path("rdf/source.nt")
    target_nt_path = get_test_data_path("rdf/target.nt")
    er_json_path = get_test_data_path("json/er.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_source_nt = Data(source_nt_path, DataFormat.RDF_NTRIPLES)
    data_target_nt = Data(target_nt_path, DataFormat.RDF_NTRIPLES)
    data_er_json = Data(er_json_path, DataFormat.ER_JSON)
    data_output = Data(output_file.name, DataFormat.RDF_NTRIPLES)

    report = union_matched_rdf.run(
        [data_source_nt, data_target_nt, data_er_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 8203

    with open(output_file.name, 'r') as f:
        print(f.read())
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_union_matched_rdf_combined():
    source_nt_path = get_test_data_path("rdf/source.nt")
    target_nt_path = get_test_data_path("rdf/target.nt")
    #TODO - different er.json's
    er_json1_path = get_test_data_path("json/er.json")
    er_json2_path = get_test_data_path("json/er.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_source_nt = Data(source_nt_path, DataFormat.RDF_NTRIPLES)
    data_target_nt = Data(target_nt_path, DataFormat.RDF_NTRIPLES)
    data_er_json1 = Data(er_json1_path, DataFormat.ER_JSON)
    data_er_json2 = Data(er_json2_path, DataFormat.ER_JSON)
    data_output = Data(output_file.name, DataFormat.RDF_NTRIPLES)

    report = union_matched_rdf_combined.run(
        [data_source_nt, data_target_nt, data_er_json1, data_er_json2],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 8203

    with open(output_file.name, 'r') as f:
        print(f.read())
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir


def test_fusion_first_value():
    source_nt_path = get_test_data_path("rdf/source.nt")
    target_nt_path = get_test_data_path("rdf/target.nt")
    er_json1_path = get_test_data_path("json/er.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_source_nt = Data(source_nt_path, DataFormat.RDF_NTRIPLES)
    data_target_nt = Data(target_nt_path, DataFormat.RDF_NTRIPLES)
    data_er_json1 = Data(er_json1_path, DataFormat.ER_JSON)
    data_output = Data(output_file.name, DataFormat.RDF_NTRIPLES)

    report = fusion_first_value.run(
        [data_source_nt, data_target_nt, data_er_json1],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 4047
    with open(output_file.name, 'r') as f:
        content = f.read()
        print(content)

    assert "<http://example.org/library/book1>" in content, \
        "'<http://example.org/library/book1>' not found!"

    assert "<http://example.org/bookstore/itemA>" not in content, \
        "'<http://example.org/bookstore/itemA>' found!"


    assert "<http://example.org/library/book2>" in content, \
        "'<http://example.org/library/book2>' not found!"

    assert "<http://example.org/bookstore/itemB>" not in content, \
        "'<http://example.org/bookstore/itemB>' found!"


    assert "<http://example.org/library/book3>" in content, \
        "'<http://example.org/library/book3>' not found!"

    assert "<http://example.org/bookstore/itemC>" not in content, \
        "'<http://example.org/bookstore/itemC>' found!"


    assert "<http://example.org/library/book4>" in content, \
        "'<http://example.org/library/book4>' not found!"

    assert "<http://example.org/bookstore/itemD>" not in content, \
        "'<http://example.org/bookstore/itemD>' found!"

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir


def test_select_first_value():
    source_nt_path = get_test_data_path("rdf/source.nt")
    target_nt_path = get_test_data_path("rdf/target.nt")
    er_json1_path = get_test_data_path("json/er.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")

    data_source_nt = Data(source_nt_path, DataFormat.RDF_NTRIPLES)
    data_target_nt = Data(target_nt_path, DataFormat.RDF_NTRIPLES)
    data_er_json1 = Data(er_json1_path, DataFormat.ER_JSON)
    data_output = Data(output_file.name, DataFormat.RDF_NTRIPLES)

    report = select_first_value.run(
        [data_source_nt, data_target_nt, data_er_json1],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 4047
    with open(output_file.name, 'r') as f:
        content = f.read()
        print(content)

    assert "<http://example.org/library/book1>" in content, \
        "'<http://example.org/library/book1>' not found!"

    assert "<http://example.org/bookstore/itemA>" not in content, \
        "'<http://example.org/bookstore/itemA>' found!"


    assert "<http://example.org/library/book2>" in content, \
        "'<http://example.org/library/book2>' not found!"

    assert "<http://example.org/bookstore/itemB>" not in content, \
        "'<http://example.org/bookstore/itemB>' found!"


    assert "<http://example.org/library/book3>" in content, \
        "'<http://example.org/library/book3>' not found!"

    assert "<http://example.org/bookstore/itemC>" not in content, \
        "'<http://example.org/bookstore/itemC>' found!"


    assert "<http://example.org/library/book4>" in content, \
        "'<http://example.org/library/book4>' not found!"

    assert "<http://example.org/bookstore/itemD>" not in content, \
        "'<http://example.org/bookstore/itemD>' found!"

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir


def test_label_based_entity_linker():

    kg_path = get_test_data_path("rdf/source.nt")
    te_json_path = get_test_data_path("json/te.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".te.json")

    data_kg = Data(kg_path, DataFormat.RDF_NTRIPLES)
    data_te_json = Data(te_json_path, DataFormat.TE_JSON)
    data_output = Data(output_file.name, DataFormat.TE_JSON)

    report = label_based_entity_linker.run(
        [data_kg, data_te_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 2547

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir