from typing import Dict, Any
from kgpipe.common import Data, DataFormat, Registry, KgTask
from pathlib import Path

def genie_text_extraction_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    from param_opti.tasks.genie_lib import genie_task_docker, genie_exchange
    
    # Ensure parent directory exists for the TE JSON output path
    outputs["output"].path.parent.mkdir(parents=True, exist_ok=True)

    input_path: Path = inputs["input"].path
    final_te_output: Data = outputs["output"]

    # 1) Produce intermediate OpenIE JSON (file or directory)
    if input_path.is_dir():
        genie_out_path = final_te_output.path.parent / f"{final_te_output.path.stem}_corenlp_openie_out"
    else:
        genie_out_path = final_te_output.path.parent / f"{final_te_output.path.stem}_corenlp_openie.json"

    genie_outpit = {"output": Data(genie_out_path, DataFormat.OPENIE_JSON)}
    genie_task_docker({"input": inputs["input"]}, genie_outpit)

    # 2) Convert OpenIE JSON → TE JSON (final output)
    genie_exchange({"input": genie_outpit["output"]}, {"output": final_te_output})


genie_text_extraction_task = KgTask(
    name="genie_text_extraction",
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.TE_JSON},
    function=genie_text_extraction_function,
    description="Extract text using Genie"
)