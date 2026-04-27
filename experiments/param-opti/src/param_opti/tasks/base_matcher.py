from kgpipe.common import TaskInput, TaskOutput, Data, DataFormat, Registry, BasicTaskCategoryCatalog
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType, ConfigurationProfile
from kgpipe.common.model.task import KgTask

def relation_matcher_label_alias_embedding_transformer_function(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    """
    Match relations using a base transformer model.
    """
    pass
    # relation_text = inputs["relation_text"]
    # relation_matcher = RelationMatcherBaseTransformer(relation_text)
    # relation_matcher.match()
    # outputs["relation_matcher"] = relation_matcher.relation_matcher


relation_matcher_label_alias_embedding_transformer_task = KgTask(
    name="relation_matcher_label_alias_embedding_transformer",
    function=relation_matcher_label_alias_embedding_transformer_function,
    input_spec={"source": DataFormat.RDF, "target": DataFormat.RDF},
    output_spec={"output": DataFormat.AGREEMENTMAKER_RDF},
    config_spec=ConfigurationDefinition(
        name="relation_matcher_label_alias_embedding_transformer",
        parameters=[
            Parameter(name="model_name", native_keys=["--model-name"], datatype=ParameterType.string, default_value="sentence-transformers/all-MiniLM-L6-v2", required=True, allowed_values=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]),
            Parameter(name="similarity_threshold", native_keys=["--similarity-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
    )
)
def entity_matcher_label_alias_embedding_transformer_function(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    """
    Match entities using a base transformer model.
    """
    pass
    # entity_text = inputs["entity_text"]
    # entity_matcher = EntityMatcherBaseTransformer(entity_text)
    # entity_matcher.match()
    # outputs["entity_matcher"] = entity_matcher.entity_matcher

entity_matcher_label_alias_embedding_transformer_task = KgTask(
    name="entity_matcher_label_alias_embedding_transformer",
    function=entity_matcher_label_alias_embedding_transformer_function,
    input_spec={"source": DataFormat.RDF, "target": DataFormat.RDF},
    output_spec={"output": DataFormat.AGREEMENTMAKER_RDF},
    config_spec=ConfigurationDefinition(
        name="entity_matcher_label_alias_embedding_transformer",
        parameters=[
            Parameter(name="model_name", native_keys=["--model-name"], datatype=ParameterType.string, default_value="sentence-transformers/all-MiniLM-L6-v2", required=True, allowed_values=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]),
            Parameter(name="similarity_threshold", native_keys=["--similarity-threshold"], datatype=ParameterType.number, default_value=0.5, required=True, allowed_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
    )
)