from typing import Dict, Any
from kgpipe.common import Data, DataFormat, Registry, KgTask
from kgpipe.common.model.configuration import ConfigurationDefinition, ConfigurationProfile, Parameter, ParameterType
from pathlib import Path

def spotlight_entity_linking_function(inputs: Dict[str, Data], outputs: Dict[str, Data], config: ConfigurationProfile   ):
    from param_opti.tasks.spotlight_lib import dbpedia_spotlight_ner_nel, dbpedia_spotlight_exchange

    # Ensure parent directory exists for the TE JSON output path
    outputs["output"].path.parent.mkdir(parents=True, exist_ok=True)

    input_path: Path = inputs["input"].path
    final_te_output: Data = outputs["output"]

    # 1) Produce intermediate OpenIE JSON (file or directory)
    if input_path.is_dir():
        spotlight_out_path = final_te_output.path.parent / f"{final_te_output.path.stem}_corenlp_openie_out"
    else:
        spotlight_out_path = final_te_output.path.parent / f"{final_te_output.path.stem}_corenlp_openie.json"

    spotlight_out = {"output": Data(spotlight_out_path, DataFormat.OPENIE_JSON)}
    if not spotlight_out_path.exists():
        dbpedia_spotlight_ner_nel({"input": inputs["input"]}, spotlight_out)

    # 2) Convert OpenIE JSON → TE JSON (final output)
    dbpedia_spotlight_exchange({"input": spotlight_out["output"]}, {"output": final_te_output}, config.get_parameter_value("similarity_threshold"))



spotlight_entity_linking_task = KgTask(
    name="spotlight_entity_linking",
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.TE_JSON},
    function=spotlight_entity_linking_function,
    description="Link entities using Spotlight",
    config_spec=ConfigurationDefinition(
        name="spotlight_entity_linking",
        parameters=[
            Parameter(name="similarity_threshold", native_keys=["--similarity-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
    )
)