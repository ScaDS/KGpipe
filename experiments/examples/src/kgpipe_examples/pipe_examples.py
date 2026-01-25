# Example pipeline definition

"""
import kgpipe.common.api

discover 
"""

from kgpipe.common import KgPipe, KgTask, Data
from .task_examples import pipe_task_python, pipe_task_docker, pipe_task_remote
from .config import ExtendedFormats

import tempfile
import os
import shutil

def pipe_example():

    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=os.path.join(tmp_data_dir, "input.special_in"), format=ExtendedFormats.SPECIAL_IN)
    output_data = Data(path=os.path.join(tmp_data_dir, "output.special_kg"), format=ExtendedFormats.SPECIAL_KG)
    
    tasks = [pipe_task_python, pipe_task_docker, pipe_task_remote]

    pipe = KgPipe(
        tasks=tasks,
        seed=input_data,
        data_dir=tmp_data_dir
    )
    pipe.build(source=input_data, result=output_data)
    pipe.run()

    # remove tmp data dir
    shutil.rmtree(tmp_data_dir)
