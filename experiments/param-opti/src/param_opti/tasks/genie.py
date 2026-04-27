from typing import Dict, Any
from kgpipe.common import Data, DataFormat, Registry, KgTask


def genie_text_extraction_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    pass

genie_text_extraction_task = KgTask(
    name="genie_text_extraction",
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.TE_JSON},
    function=genie_text_extraction_function,
    description="Extract text using Genie"
)