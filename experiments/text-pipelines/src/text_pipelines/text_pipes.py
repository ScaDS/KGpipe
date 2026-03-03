# Example pipeline definition

"""
import kgpipe.common.api

discover 
"""

from kgpipe.common import KgPipe, Data, DataFormat

import tempfile
import shutil

def run_pipe(input_path: str, output_path: str, tasks: list):
    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=input_path, format=DataFormat.TEXT)
    output_data = Data(path=output_path, format=DataFormat.TE_JSON)

    pipe = KgPipe(
        tasks=tasks,
        seed=input_data,
        data_dir=tmp_data_dir
    )

    pipe.build(source=input_data, result=output_data)
    pipe.run()

    # remove tmp data dir
    shutil.rmtree(tmp_data_dir)