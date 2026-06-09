import csv
import json
import os
import re
from typing import Dict, Any

from rdflib import Graph, URIRef, Literal, BNode

from kgpipe.common import TaskInput, TaskOutput, DataFormat, get_docker_volume_bindings, remap_data_path_for_container, \
    Data
from kgpipe.common.registry import Registry
from kgpipe.execution import docker_client


def test_ontology_matching_sampled():
    ONTOLOGY=""
    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=input_path, format=DataFormat.TEXT)
    output_data = Data(path=output_path, format=DataFormat.ANY)
    
    tasks = [openie6_task_docker]

    pipe = KgPipe(
        tasks=tasks,
        seed=input_data,
        data_dir=tmp_data_dir
    )

    pipe.build(source=input_data, result=output_data)
    pipe.run()

    # remove tmp data dir
    shutil.rmtree(tmp_data_dir)


    pass

def test_ontology_matching_sampled_single():
    ONTOLOGY=""
    pass

def test_ontology_matching_embedd():
    ONTOLOGY=""
    pass

def test_entity_matching_embedd():
    pass

def test_entity_clustering():
    pass