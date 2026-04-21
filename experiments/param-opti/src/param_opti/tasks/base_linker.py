from kgpipe.common import TaskInput, TaskOutput, Data, DataFormat, KgTask
from kgpipe.common.model.configuration import ConfigurationProfile, ConfigurationDefinition, Parameter, ParameterType

def relation_linker_label_alias_embedding_transformer_function(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    """
    Link relations using a base transformer model.
    """
    pass
    # relation_text = inputs["relation_text"]
    # relation_linker = RelationLinkerBaseTransformer(relation_text)
    # relation_linker.link()
    # outputs["relation_link"] = relation_linker.relation_link

relation_linker_label_alias_embedding_transformer_task = KgTask(
    name="relation_linker_label_alias_embedding_transformer",
    function=relation_linker_label_alias_embedding_transformer_function,
    input_spec={"source": DataFormat.RDF, "target": DataFormat.RDF},
    output_spec={"output": DataFormat.RDF},
    config_spec=ConfigurationDefinition(
        name="relation_linker_label_alias_embedding_transformer",
        parameters=[
            Parameter(name="model_name", native_keys=["--model-name"], datatype=ParameterType.string, default_value="sentence-transformers/all-MiniLM-L6-v2", required=True, allowed_values=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]),
            Parameter(name="similarity_threshold", native_keys=["--similarity-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
    )
)

def entity_linker_label_alias_embedding_transformer_function(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    """
    Link entities using a base transformer model.
    """
    pass
    # entity_text = inputs["entity_text"]
    # entity_linker = EntityLinkerBaseTransformer(entity_text)
    # entity_linker.link()
    # outputs["entity_link"] = entity_linker.entity_link

entity_linker_label_alias_embedding_transformer_task = KgTask(
    name="entity_linker_label_alias_embedding_transformer",
    function=entity_linker_label_alias_embedding_transformer_function,
    input_spec={"source": DataFormat.RDF, "target": DataFormat.RDF},
    output_spec={"output": DataFormat.RDF},
    config_spec=ConfigurationDefinition(
        name="entity_linker_label_alias_embedding_transformer",
        parameters=[
            Parameter(name="model_name", native_keys=["--model-name"], datatype=ParameterType.string, default_value="sentence-transformers/all-MiniLM-L6-v2", required=True, allowed_values=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]),
            Parameter(name="similarity_threshold", native_keys=["--similarity-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
    )
)

