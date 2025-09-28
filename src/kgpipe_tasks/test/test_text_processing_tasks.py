from kgpipe.common import Data, DataFormat
from kgpipe_tasks.text_processing.text_extraction.corenlp_extraction import corenlp_openie_extraction
from kgpipe_tasks.text_processing.entity_linking.spotlight_entity_linking import dbpedia_spotlight_ner_nel
from kgpipe_tasks.text_processing.entity_linking.falcon_entity_linking import falcon_ner_nel_rl
from kgpipe_tasks.text_processing.relation_match import label_alias_embedding_rl
from . import get_test_data_path

import pytest
import tempfile
import os
import json

def test_dbpedia_spotlight_ner_nel():
    text_path = get_test_data_path("text/source.text/The_Hobbit.txt")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")

    data_text = Data(text_path, DataFormat.TEXT)
    data_output = Data(output_file.name, DataFormat.SPOTLIGHT_JSON)

    report = dbpedia_spotlight_ner_nel.run(
        [data_text],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 4678

    with open(output_file.name, encoding="utf-8") as f:
        content = f.read()

    assert "http://dbpedia.org/resource/The_Hobbit" in content, \
        f"Did not link 'The_Hobbit'!"

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_falcon_ner_nel_rl():
    text_path = get_test_data_path("text/source.text/The_Hobbit.txt")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")

    data_text = Data(text_path, DataFormat.TEXT)
    data_output = Data(output_file.name, DataFormat.FALCON_JSON)

    report = falcon_ner_nel_rl.run(
        [data_text],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 2085

    with open(output_file.name, encoding="utf-8") as f:
        content = f.read()

    assert "http://www.wikidata.org/entity/Q74229" in content, \
        f"Did not link 'The_Hobbit'!"

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

def test_corenlp_openie_extraction():
    text_path = get_test_data_path("text/source.text/The_Hobbit.txt")

    output_dir = tempfile.mkdtemp()

    data_text = Data(text_path, DataFormat.TEXT)
    data_output = Data(output_dir, DataFormat.OPENIE_JSON)

    report = corenlp_openie_extraction.run(
        [data_text],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]

    assert os.path.getsize(files[0]) == 77752
    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

@pytest.mark.skip(reason="Missing Task Definition")
def test_rebel_extraction():
    pass

def test_label_alias_embedding_rl():

    te_json_path = get_test_data_path("json/te.json")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".te.json")

    data_te_json = Data(te_json_path, DataFormat.TE_JSON)
    data_output = Data(output_file.name, DataFormat.TE_JSON)

    report = label_alias_embedding_rl.run(
        [data_te_json],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    assert os.path.getsize(output_file.name) == 769

    print(json.dumps(report.__dict__, indent=4, default=str))

    # TODO delete output file/dir (requires correct permissions)
    #   we can use another docker call to delete the file/dir

@pytest.mark.skip(reason="Missing Task Definition")
def test_corenlp_kbp_extraction():
    pass