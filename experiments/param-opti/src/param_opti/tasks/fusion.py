from kgpipe.common.model.configuration import ConfigurationProfile
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType
from kgpipe.common.models import TaskInput, TaskOutput, KgTask, DataFormat

def fusion_first_value_function(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    # touch output file
    from param_opti.tasks.fusion_lib import fusion_first_value
    # TODO remove thresholds as they are applied by the matchers 
    fusion_first_value(inputs, outputs, entity_matching_threshold=0.0, relation_matching_threshold=0.0, ontology_path=config.get_parameter_value("ontology_path"))

fusion_first_value_task = KgTask(
        name="fusion_first_value",
        function=fusion_first_value_function,
        input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES, "matches1": DataFormat.ER_JSON},
        output_spec={"output": DataFormat.RDF_NTRIPLES},
        config_spec=ConfigurationDefinition(
            name="fusion_first_value",
            parameters=[
                # ontology path
                Parameter(name="ontology_path", native_keys=["--ontology-path"], datatype=ParameterType.string, default_value="", required=True),
            ]
        )
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