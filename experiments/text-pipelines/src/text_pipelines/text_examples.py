import csv
import json
import os
import tempfile
from pathlib import Path

from kgpipe.common import Data, DataFormat

docker_task_input_path = Path("experiments/text-pipelines/test/Titanic.txt")

def _delete_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            print(e)

# Docker Task Test
def test_openie6_docker_task():
    from text_pipelines.text_tasks import openie6_task_docker
    pass

def test_graphene_docker_task():
    from text_pipelines.text_tasks import graphene_task_docker

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.nt")

    data_source_rdf = Data(docker_task_input_path, DataFormat.TEXT)
    data_output = Data(task_output_path, DataFormat.RDF_NTRIPLES)

    report = graphene_task_docker.run(
        [data_source_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    assert len(lines) == 583

    _delete_file(task_output_path)

def test_minie_docker_task():
    from text_pipelines.text_tasks import minie_task_docker

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.any")

    data_source_rdf = Data(docker_task_input_path, DataFormat.TEXT)
    data_output = Data(task_output_path, DataFormat.ANY)

    report = minie_task_docker.run(
        [data_source_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    assert len(lines) == 208

    _delete_file(task_output_path)

def test_imojie_docker_task():
    from text_pipelines.text_tasks import imojie_task_docker

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.any")

    data_source_rdf = Data(docker_task_input_path, DataFormat.TEXT)
    data_output = Data(task_output_path, DataFormat.ANY)

    report = imojie_task_docker.run(
        [data_source_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    assert len(lines) == 32

    _delete_file(task_output_path)

def test_genie_docker_task():
    from text_pipelines.text_tasks import genie_task_docker

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.json")

    data_source_rdf = Data(docker_task_input_path, DataFormat.TEXT)
    data_output = Data(task_output_path, DataFormat.JSON)

    report = genie_task_docker.run(
        [data_source_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    assert len(lines) == 12

    _delete_file(task_output_path)


# Exchange Task Test
def test_graphene_output_to_json_te():
    from text_pipelines.text_tasks import graphene_nt_exchange

    exchange_task_input_path = Path("experiments/text-pipelines/test/graphene_output.nt")

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.te.json")

    data_source_rdf = Data(exchange_task_input_path, DataFormat.RDF_NTRIPLES)
    data_output = Data(task_output_path, DataFormat.TE_JSON)

    report = graphene_nt_exchange.run(
        [data_source_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    assert "triples" in data
    assert "chains" in data

    assert len(data["triples"]) == 1

    triple = data["triples"][0]

    assert "subject" in triple
    assert "predicate" in triple
    assert "object" in triple

    assert triple["subject"]["surface_form"] == "This"
    assert triple["predicate"]["surface_form"] == "is"
    assert triple["object"]["surface_form"] == "a simple test"

    _delete_file(task_output_path)

def test_minie_exchange():
    from text_pipelines.text_tasks import minie_exchange

    exchange_task_input_path = Path("experiments/text-pipelines/test/minie_output.txt")

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.any")

    data_source_rdf = Data(exchange_task_input_path, DataFormat.TEXT)
    data_output = Data(task_output_path, DataFormat.TE_JSON)

    report = minie_exchange.run(
        [data_source_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    assert "triples" in data
    assert "chains" in data

    assert len(data["triples"]) == 1

    triple = data["triples"][0]

    assert "subject" in triple
    assert "predicate" in triple
    assert "object" in triple

    assert triple["subject"]["surface_form"] == "This"
    assert triple["predicate"]["surface_form"] == "is"
    assert triple["object"]["surface_form"] == "simple test"

    _delete_file(task_output_path)

def test_imojie_exchange():
    from text_pipelines.text_tasks import imojie_exchange

    exchange_task_input_path = Path("experiments/text-pipelines/test/imojie_output.txt")

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.any")

    data_source_rdf = Data(exchange_task_input_path, DataFormat.TEXT)
    data_output = Data(task_output_path, DataFormat.TE_JSON)

    report = imojie_exchange.run(
        [data_source_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    assert "triples" in data
    assert "chains" in data

    assert len(data["triples"]) == 1

    triple = data["triples"][0]

    assert "subject" in triple
    assert "predicate" in triple
    assert "object" in triple

    assert triple["subject"]["surface_form"] == "This"
    assert triple["predicate"]["surface_form"] == "is"
    assert triple["object"]["surface_form"] == "a simple test"

    _delete_file(task_output_path)

def test_genie_exchange():
    from text_pipelines.text_tasks import genie_exchange

    exchange_task_input_path = Path("experiments/text-pipelines/test/genie_output.json")

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.te.json")

    data_source_rdf = Data(exchange_task_input_path, DataFormat.JSON)
    data_output = Data(task_output_path, DataFormat.TE_JSON)

    report = genie_exchange.run(
        [data_source_rdf],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    assert "triples" in data
    assert "chains" in data

    assert len(data["triples"]) == 9

    triple = data["triples"][0]

    assert "subject" in triple
    assert "predicate" in triple
    assert "object" in triple

    _delete_file(task_output_path)


def test_te_json_exchange():
    from text_pipelines.text_tasks import te_json_triple_exchange

    exchange_task_input_path = Path("experiments/text-pipelines/test/test.te.json")

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.csv")

    data_source = Data(exchange_task_input_path, DataFormat.TE_JSON)
    data_output = Data(task_output_path, DataFormat.CSV)

    report = te_json_triple_exchange.run(
        [data_source],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == ["subject", "predicate", "object"], "Wrong header"

    assert len(rows) == 4, "Wrong number of rows"

    assert rows[1] == [
        "This",
        "is",
        "of the estimated 2,224 passengers aboard"
    ], "Wrong line 1"

    _delete_file(task_output_path)

# Pipe Tests
from text_pipelines.text_pipes import run_pipe

def test_openie6_pipe():
    from text_pipelines.text_tasks import openie6_task_docker

    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.te.json")

    run_pipe(str(docker_task_input_path), output_path, [openie6_task_docker])

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    _delete_file(output_path)

def test_graphene_pipe():
    from text_pipelines.text_tasks import graphene_task_docker, graphene_nt_exchange

    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.te.json")

    run_pipe(str(docker_task_input_path), output_path, [graphene_task_docker, graphene_nt_exchange])

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    _delete_file(output_path)

def test_minie_pipe():
    from text_pipelines.text_tasks import minie_task_docker, minie_exchange

    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.te.json")

    run_pipe(str(docker_task_input_path), output_path, [minie_task_docker, minie_exchange])

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    _delete_file(output_path)

def test_imojie_pipe():
    from text_pipelines.text_tasks import imojie_task_docker, imojie_exchange

    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.te.json")

    run_pipe(str(docker_task_input_path), output_path, [imojie_task_docker, imojie_exchange])

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    _delete_file(output_path)

def test_genie_pipe():
    from text_pipelines.text_tasks import genie_task_docker, genie_exchange

    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.te.json")

    run_pipe(str(docker_task_input_path), output_path, [genie_task_docker, genie_exchange])

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    _delete_file(output_path)
