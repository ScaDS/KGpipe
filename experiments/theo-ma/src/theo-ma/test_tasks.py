import csv
import json
import os
import tempfile
from pathlib import Path

from kgpipe.common import Data, DataFormat
def _delete_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            print(e)

# Docker Task Test

def test_valentine_docker_task():
    from text_tasks import valentine_task_docker

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir, "output.json")

    path1 = Path(
        "/home/theo/Uni/Masterarbeit/datasets/valentine-datasets/Valentine-datasets/TPC-DI/Unionable/prospect_horizontal_0_ac1_av/prospect_horizontal_0_ac1_av_source.csv")
    path2 = Path(
        "/home/theo/Uni/Masterarbeit/datasets/valentine-datasets/Valentine-datasets/TPC-DI/Unionable/prospect_horizontal_0_ac1_av/prospect_horizontal_0_ac1_av_target.csv")

    path11 = Data(path1, DataFormat.CSV)
    path12 = Data(path2, DataFormat.CSV)
    data_output = Data(task_output_path, DataFormat.ER_JSON)

    report = valentine_task_docker.run(
        [path11, path12],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

def test_pyjedai_docker_task():
    from text_tasks import pyjedai_task_docker

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir)

    path11 = Data("/home/theo/Uni/Masterarbeit/datasets/pyjedai-datasets/ccer/D1/rest1.csv", DataFormat.CSV)
    path12 = Data("/home/theo/Uni/Masterarbeit/datasets/pyjedai-datasets/ccer/D1/rest2.csv", DataFormat.CSV)
    gt = Data("/home/theo/Uni/Masterarbeit/datasets/pyjedai-datasets/ccer/D1/gt.csv", DataFormat.CSV)
    data_output = Data(task_output_path, DataFormat.ER_JSON)

    report = pyjedai_task_docker.run(
        [path11, path12, gt],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(lines)


def test_paris_docker_task():
    from text_tasks import paris_task_docker

    output_dir = tempfile.mkdtemp()
    task_output_path = os.path.join(output_dir)

    path11 = Data("/home/theo/Uni/Masterarbeit/datasets/paris-datasets/restaurant1.rdf", DataFormat.RDF)
    path12 = Data("/home/theo/Uni/Masterarbeit/datasets/paris-datasets/restaurant2.rdf", DataFormat.RDF)
    data_output = Data(task_output_path, DataFormat.ER_JSON)

    report = paris_task_docker.run(
        [path11, path12],
        [data_output],
        stable_files_override=True
    )

    assert report.status == "success"

    with open(task_output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(lines)
