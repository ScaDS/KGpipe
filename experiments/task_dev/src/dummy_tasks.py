from kgpipe.common.registry import Registry
from kgpipe.common.models import DataFormat, Data

@Registry.task(input_spec={"source": DataFormat.RDF_NTRIPLES}, output_spec={"output": DataFormat.RDF_NTRIPLES})
def dummy_task(inputs: dict[str, Data], outputs: dict[str, Data]):
    print(inputs)
    print(outputs)
    