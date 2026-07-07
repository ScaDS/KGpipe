from typing import Dict

from pathlib import Path
from kgpipe.common import Data, DataFormat, Registry, KgTask
from kgpipe.common.model.configuration import ConfigurationDefinition


def corenlp_text_extraction_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    from param_opti.tasks.corenlp_lip import corenlp_openie_extraction, corenlp_exchange

    # Ensure parent directory exists for the TE JSON output path
    outputs["output"].path.parent.mkdir(parents=True, exist_ok=True)

    input_path: Path = inputs["input"].path
    final_te_output: Data = outputs["output"]

    # 1) Produce intermediate OpenIE JSON (file or directory)
    if input_path.is_dir():
        openie_out_path = final_te_output.path.parent / f"{final_te_output.path.stem}_corenlp_openie_out"
    else:
        openie_out_path = final_te_output.path.parent / f"{final_te_output.path.stem}_corenlp_openie.json"

    openie_output = {"output": Data(openie_out_path, DataFormat.OPENIE_JSON)}
    corenlp_openie_extraction({"input": inputs["input"]}, openie_output)

    # 2) Convert OpenIE JSON → TE JSON (final output)
    corenlp_exchange({"input": openie_output["output"]}, {"output": final_te_output})


corenlp_text_extraction_task = KgTask(
    name="corenlp_text_extraction",
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.TE_JSON},
    function=corenlp_text_extraction_function,
    description="Extract text using CoreNLP"
)