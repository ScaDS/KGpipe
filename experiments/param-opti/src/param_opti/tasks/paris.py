from kgpipe.common import TaskInput, TaskOutput, KgTask, DataFormat
from kgpipe.common.model.configuration import ConfigurationProfile, ConfigurationDefinition, Parameter, ParameterType



def paris_entity_alignment_function(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    """
    matches entities between two RDF graphs
    """
    # touch output file
    print(f"paris_entity_alignment_function: {outputs['output'].path}")
    outputs["output"].path.touch()

paris_entity_alignment_task = KgTask(
    name="paris_entity_alignment",
    function=paris_entity_alignment_function,
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    config_spec=ConfigurationDefinition(
        name="paris_entity_alignment",
        parameters=[
            Parameter(name="entity_matching_threshold", native_keys=["--entity-matching-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
    )
)

def paris_graph_alignment_function(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    """
    matches both entities and relations between two RDF graphs
    """
    # touch output file
    outputs["output"].path.touch()

paris_graph_alignment_task = KgTask(
    name="paris_graph_alignment",
    function=paris_graph_alignment_function,
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    config_spec=ConfigurationDefinition(
        name="paris_graph_alignment",
        parameters=[
            Parameter(name="entity_matching_threshold", native_keys=["--entity-matching-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            Parameter(name="relation_matching_threshold", native_keys=["--relation-matching-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
    )
)