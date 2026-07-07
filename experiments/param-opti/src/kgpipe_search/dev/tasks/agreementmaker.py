from kgpipe.common import Data, DataFormat, KgTask, Registry, TaskInput, TaskOutput, BasicTaskCategoryCatalog

@Registry.task(
    input_spec={"source": DataFormat.RDF, "target": DataFormat.RDF},
    output_spec={"output": DataFormat.AGREEMENTMAKER_RDF},
    description="Perform entity matching using AgreementMaker",
    category=[BasicTaskCategoryCatalog.entity_matching]
)
def entity_matching_aggrement_maker(inputs: TaskInput, outputs: TaskOutput):
    """Perform entity matching using AgreementMaker."""
    source_data = inputs["source"]
    target_data = inputs["target"]
    output_data = outputs["output"]
    return output_data