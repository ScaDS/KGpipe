# Example pipeline definition

"""
import kgpipe.common.api

discover 
"""

from kgpipe.common import KgPipe, Data, DataFormat

import tempfile

def run_pipe(input_path: str, output_path: str, seed_path: str, input_format, output_format, tasks: list):
    tmp_data_dir = tempfile.mkdtemp()
    input_data = Data(path=input_path, format=input_format)
    output_data = Data(path=output_path, format=output_format)

    seed_data = Data(path=seed_path, format=DataFormat.RDF_TTL)

    pipe = KgPipe(
        tasks=tasks,
        seed=seed_data,
        data_dir=tmp_data_dir
    )

    pipe.build(source=input_data, result=output_data)
    pipe.run()
