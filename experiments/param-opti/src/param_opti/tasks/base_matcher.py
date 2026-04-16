from kgpipe.common import TaskInput, TaskOutput, Data, DataFormat, Registry, BasicTaskCategoryCatalog
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType

@Registry.task(
    input_spec={"source": DataFormat.RDF, "target": DataFormat.RDF},
    output_spec={"output": DataFormat.AGREEMENTMAKER_RDF},
    description="Perform entity matching using AgreementMaker",
    category=[BasicTaskCategoryCatalog.entity_matching],
    config_spec=ConfigurationDefinition(
        parameters=[
            Parameter(name="model_name", type=ParameterType.STRING, default="sentence-transformers/all-MiniLM-L6-v2"),
            Parameter(name="similarity_threshold", type=ParameterType.NUMBER, default=0.5),
        ]
    )
)
def relation_matcher_label_alias_embedding_transformer(inputs: TaskInput, outputs: TaskOutput):
    """
    Match relations using a base transformer model.
    """
    pass
    # relation_text = inputs["relation_text"]
    # relation_matcher = RelationMatcherBaseTransformer(relation_text)
    # relation_matcher.match()
    # outputs["relation_matcher"] = relation_matcher.relation_matcher

@Registry.task(
    input_spec={"source": DataFormat.RDF, "target": DataFormat.RDF},
    output_spec={"output": DataFormat.AGREEMENTMAKER_RDF},
    description="Perform entity matching using AgreementMaker",
    category=[BasicTaskCategoryCatalog.entity_matching],
    config_spec=ConfigurationDefinition(
        parameters=[
            Parameter(name="model_name", type=ParameterType.STRING, default="sentence-transformers/all-MiniLM-L6-v2"),
            Parameter(name="similarity_threshold", type=ParameterType.NUMBER, default=0.5),
        ]
    )
)
def entity_matcher_label_alias_embedding_transformer(inputs: TaskInput, outputs: TaskOutput):
    """
    Match entities using a base transformer model.
    """
    pass
    # entity_text = inputs["entity_text"]
    # entity_matcher = EntityMatcherBaseTransformer(entity_text)
    # entity_matcher.match()
    # outputs["entity_matcher"] = entity_matcher.entity_matcher