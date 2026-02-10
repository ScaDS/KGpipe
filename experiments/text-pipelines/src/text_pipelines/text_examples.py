import os
import tempfile
from pathlib import Path

from kgpipe.common import Data, DataFormat

task_input_path = Path("experiments/text-pipelines/test/Titanic.txt")

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

    data_source_rdf = Data(task_input_path, DataFormat.TEXT)
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

    data_source_rdf = Data(task_input_path, DataFormat.TEXT)
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

    data_source_rdf = Data(task_input_path, DataFormat.TEXT)
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


# Exchange Task Test
def test_graphene_output_to_json_te():
    from text_pipelines.text_tasks import graphene_nt_exchange
    pass

def test_minie_exchange():
    from text_pipelines.text_tasks import minie_exchange
    pass

def test_imojie_exchange():
    from text_pipelines.text_tasks import imojie_exchange
    pass


# Pipe Tests
def test_openie6_pipe():
    from text_pipelines.text_pipes import openie6_pipe
    openie6_pipe("input.txt", "output.json")

def test_graphene_pipe():
    from text_pipelines.text_pipes import graphene_pipe
    graphene_pipe("input.txt", "output.json")

def test_minie_pipe():
    from text_pipelines.text_pipes import minie_pipe
    minie_pipe("input.txt","output.json")

def test_imojie_pipe():
    from text_pipelines.text_pipes import imojie_pipe
    imojie_pipe("input.txt","output.json")