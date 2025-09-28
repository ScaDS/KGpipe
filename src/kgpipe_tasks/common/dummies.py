from kgpipe.common.models import DataFormat, Data
from kgpipe.common.models import KgTask
from typing import Dict
from kgpipe.execution.config import GLOBAL_STATE

def dummy_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    print(GLOBAL_STATE)
    pass

dummy_task = KgTask(
    name="dummy",
    input_spec={"source": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    function=dummy_function
)