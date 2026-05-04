from typing import Dict, Any
from kgpipe.common import Data, DataFormat, Registry, KgTask
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType

def spotlight_entity_linking_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    from param_opti.tasks.spotlight_lib import dbpedia_spotlight_ner_nel, dbpedia_spotlight_exchange_filtered

    dbpedia_spotlight_ner_nel({"input": inputs["input"]}, {"output": outputs["output"]})
    dbpedia_spotlight_exchange_filtered({"source": outputs["output"]}, {"output": outputs["output"]})

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