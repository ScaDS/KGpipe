# Example pipeline definition

"""
import kgpipe.common.api

discover 
"""

from kgpipe.common import KgPipe, Data, DataFormat
from text_pipelines.text_tasks import openie6_task_docker, graphene_nt_exchange, graphene_task_docker, \
    minie_task_docker, minie_exchange

import tempfile
import os
import shutil

TEXT="""
Titanic is a 1997 American epic historical romance film written and directed by James Cameron. Incorporating both historical and fictional aspects, it is based on accounts of the sinking of RMS Titanic in 1912. Leonardo DiCaprio and Kate Winslet star as members of different social classes who fall in love during the ship's ill-fated maiden voyage. The ensemble cast includes Billy Zane, Kathy Bates, Frances Fisher, Bernard Hill, Jonathan Hyde, Danny Nucci, David Warner and Bill Paxton.
Cameron's inspiration came from his fascination with shipwrecks. He felt a love story interspersed with human loss would be essential to convey the emotional impact of the disaster. Production began on September 1, 1995,[8] when Cameron shot footage of the Titanic wreck. The modern scenes on the research vessel were shot on board the Akademik Mstislav Keldysh, which Cameron had used as a base when filming the wreck. Scale models, computer-generated imagery (CGI), and a reconstruction of the Titanic built at Baja Studios were used to recreate the sinking. Titanic was initially in development at 20th Century Fox, but delays and a mounting budget resulted in Fox partnering with Paramount Pictures for financial help. It was the most expensive film ever made at the time, with a production budget of $200 million. Filming took place from July 1996 to March 1997.
"""

def openie6_pipe():

    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=os.path.join(tmp_data_dir, "input.special_in"), format=DataFormat.TEXT)
    output_data = Data(path=os.path.join(tmp_data_dir, "output.txt"), format=DataFormat.TEXT)
    
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