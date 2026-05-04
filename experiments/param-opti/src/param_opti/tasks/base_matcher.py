from kgpipe.common import TaskInput, TaskOutput, DataFormat
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType, ConfigurationProfile
from kgpipe.common.model.task import KgTask

# Same as paris_graph_alignment_task / paris_entity_alignment_task:
# input_spec + output_spec as in experiments/param-opti/src/param_opti/tasks/paris.py (e.g. lines 58–59).
_ALIGNMENT_TWO_GRAPH_INPUT_SPEC = {"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES}
_ALIGNMENT_ER_JSON_OUTPUT_SPEC = {"output": DataFormat.ER_JSON}


def _embedding_config_params():
    return [
        Parameter(
            name="model_name",
            native_keys=["--model-name"],
            datatype=ParameterType.string,
            default_value="sentence-transformers/all-MiniLM-L6-v2",
            required=True,
            allowed_values=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
        ),
        Parameter(
            name="similarity_threshold",
            native_keys=["--similarity-threshold"],
            datatype=ParameterType.number,
            default_value=0.5,
            required=True,
            allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ),
    ]


def graph_alignment_label_alias_embedding_transformer_function(
    inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile
):
    """Match entities and relations between two RDF graphs (full graph alignment)."""
    from param_opti.tasks.base_matcher_lib import label_embedding_graph_alignment_match

    label_embedding_graph_alignment_match(
        inputs,
        outputs,
        model_name=config.get_parameter_value("model_name"),
        threshold=float(config.get_parameter_value("similarity_threshold")),
    )


graph_alignment_label_alias_embedding_transformer_task = KgTask(
    name="graph_alignment_label_alias_embedding_transformer",
    function=graph_alignment_label_alias_embedding_transformer_function,
    input_spec=dict(_ALIGNMENT_TWO_GRAPH_INPUT_SPEC),
    output_spec=dict(_ALIGNMENT_ER_JSON_OUTPUT_SPEC),
    config_spec=ConfigurationDefinition(
        name="graph_alignment_label_alias_embedding_transformer",
        parameters=_embedding_config_params(),
    ),
)


def entity_matcher_label_alias_embedding_transformer_function(
    inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile
):
    """Entity alignment only (subject/object URIs with rdfs:label)."""
    from param_opti.tasks.base_matcher_lib import label_embedding_entity_alignment_match

    label_embedding_entity_alignment_match(
        inputs,
        outputs,
        model_name=config.get_parameter_value("model_name"),
        threshold=float(config.get_parameter_value("similarity_threshold")),
    )


entity_matcher_label_alias_embedding_transformer_task = KgTask(
    name="entity_matcher_label_alias_embedding_transformer",
    function=entity_matcher_label_alias_embedding_transformer_function,
    input_spec=dict(_ALIGNMENT_TWO_GRAPH_INPUT_SPEC),
    output_spec=dict(_ALIGNMENT_ER_JSON_OUTPUT_SPEC),
    config_spec=ConfigurationDefinition(
        name="entity_matcher_label_alias_embedding_transformer",
        parameters=_embedding_config_params(),
    ),
)


def relation_matcher_label_alias_embedding_transformer_function(
    inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile
):
    """Relation / predicate alignment only."""
    from param_opti.tasks.base_matcher_lib import label_embedding_relation_alignment_match

    label_embedding_relation_alignment_match(
        inputs,
        outputs,
        model_name=config.get_parameter_value("model_name"),
        threshold=float(config.get_parameter_value("similarity_threshold")),
    )


relation_matcher_label_alias_embedding_transformer_task = KgTask(
    name="relation_matcher_label_alias_embedding_transformer",
    function=relation_matcher_label_alias_embedding_transformer_function,
    input_spec=dict(_ALIGNMENT_TWO_GRAPH_INPUT_SPEC),
    output_spec=dict(_ALIGNMENT_ER_JSON_OUTPUT_SPEC),
    config_spec=ConfigurationDefinition(
        name="relation_matcher_label_alias_embedding_transformer",
        parameters=_embedding_config_params(),
    ),
)
