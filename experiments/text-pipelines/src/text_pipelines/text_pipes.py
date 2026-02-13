# Example pipeline definition

"""
import kgpipe.common.api

discover 
"""

from kgpipe.common import KgPipe, Data, DataFormat
from text_pipelines.text_tasks import openie6_task_docker, graphene_nt_exchange, graphene_task_docker, \
    minie_task_docker, minie_exchange, imojie_task_docker, imojie_exchange, genie_task_docker, genie_exchange

import tempfile
import shutil

def openie6_pipe(input_path:str, output_path:str):

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

def graphene_pipe(input_path:str, output_path:str):
    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=input_path, format=DataFormat.TEXT)
    output_data = Data(path=output_path, format=DataFormat.TE_JSON)

    tasks = [graphene_task_docker, graphene_nt_exchange]

    pipe = KgPipe(
        tasks=tasks,
        seed=input_data,
        data_dir=tmp_data_dir
    )

    pipe.build(source=input_data, result=output_data)
    pipe.run()

    # remove tmp data dir
    shutil.rmtree(tmp_data_dir)

def minie_pipe(input_path:str, output_path:str):
    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=input_path, format=DataFormat.TEXT)
    output_data = Data(path=output_path, format=DataFormat.TE_JSON)

    tasks = [minie_task_docker, minie_exchange]

    pipe = KgPipe(
        tasks=tasks,
        seed=input_data,
        data_dir=tmp_data_dir
    )

    pipe.build(source=input_data, result=output_data)
    pipe.run()

    # remove tmp data dir
    shutil.rmtree(tmp_data_dir)

def imojie_pipe(input_path: str, output_path: str):
    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=input_path, format=DataFormat.TEXT)
    output_data = Data(path=output_path, format=DataFormat.TE_JSON)

    tasks = [imojie_task_docker, imojie_exchange]

    pipe = KgPipe(
        tasks=tasks,
        seed=input_data,
        data_dir=tmp_data_dir
    )

    pipe.build(source=input_data, result=output_data)
    pipe.run()

    # remove tmp data dir
    shutil.rmtree(tmp_data_dir)

def genie_pipe(input_path: str, output_path: str):
    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=input_path, format=DataFormat.TEXT)
    output_data = Data(path=output_path, format=DataFormat.TE_JSON)

    tasks = [genie_task_docker, genie_exchange]

    pipe = KgPipe(
        tasks=tasks,
        seed=input_data,
        data_dir=tmp_data_dir
    )

    pipe.build(source=input_data, result=output_data)
    pipe.run()

    # remove tmp data dir
    shutil.rmtree(tmp_data_dir)