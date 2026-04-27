from kgpipe.common import TaskInput, TaskOutput, KgTask, DataFormat, Data
from kgpipe.common.model.configuration import ConfigurationProfile, ConfigurationDefinition, Parameter, ParameterType
from pathlib import Path



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
    from param_opti.tasks.paris_lib import paris_exchange, paris_entity_matching
    entity_matching_threshold = float(config.get_parameter_value("entity_matching_threshold"))
    relation_matching_threshold = float(config.get_parameter_value("relation_matching_threshold"))

    # Ensure parent directory exists for the ER JSON output file
    outputs["output"].path.parent.mkdir(parents=True, exist_ok=True)

    # 1 produce matches in paris csv format
    matching_dir = outputs["output"].path.parent / f"{outputs['output'].path.stem}_paris_out"
    matching_output = {"output": Data(matching_dir, DataFormat.PARIS_CSV)}

    # paris_entity_matching expects {"source": ..., "kg": ...}
    paris_entity_matching({"source": inputs["source"], "kg": inputs["target"]}, matching_output)

    # 2 convert paris output dir to er.json format (file)
    paris_exchange(
        matching_output["output"].path,
        outputs["output"].path,
        entity_matching_threshold,
        relation_matching_threshold,
    )

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

def paris_ontology_matching_function(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    """
    matches ontologies between two RDF graphs
    """
    # touch output file
    outputs["output"].path.touch()

paris_ontology_matching_task = KgTask(
    name="paris_ontology_matching",
    function=paris_ontology_matching_function,
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    config_spec=ConfigurationDefinition(
        name="paris_ontology_matching",
        parameters=[
            Parameter(name="ontology_matching_threshold", native_keys=["--ontology-matching-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
    )
)