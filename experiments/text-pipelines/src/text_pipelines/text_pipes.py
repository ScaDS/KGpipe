# Example pipeline definition

"""
import kgpipe.common.api

discover 
"""

from kgpipe.common import KgPipe, Data, DataFormat
from text_pipelines.text_tasks import openie6_task_docker

import tempfile
import os
import shutil

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
