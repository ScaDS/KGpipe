from kgpipe.common.model.configuration import ConfigurationProfile
from kgpipe.common.models import TaskInput, TaskOutput, KgTask, DataFormat

def fusion_first_value_function(inputs: TaskInput, outputs: TaskOutput):
    # touch output file
    outputs["output"].path.touch()

fusion_first_value_task = KgTask(
        name="fusion_first_value",
        function=fusion_first_value_function,
        input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES, "matches": DataFormat.ER_JSON},
        output_spec={"output": DataFormat.RDF_NTRIPLES},
)

def fusion_union_function(inputs: TaskInput, outputs: TaskOutput):
    # touch output file
    outputs["output"].path.touch()

fusion_union_task = KgTask(
    name="fusion_union",
    function=fusion_union_function,
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES, "matches": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
)