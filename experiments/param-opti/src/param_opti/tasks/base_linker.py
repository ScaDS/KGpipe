from kgpipe.common import TaskInput, TaskOutput, Data, DataFormat

def relation_linker_label_alias_embedding_transformer(inputs: TaskInput, outputs: TaskOutput):
    """
    Link relations using a base transformer model.
    """
    pass
    # relation_text = inputs["relation_text"]
    # relation_linker = RelationLinkerBaseTransformer(relation_text)
    # relation_linker.link()
    # outputs["relation_link"] = relation_linker.relation_link


def entity_linker_label_alias_embedding_transformer(inputs: TaskInput, outputs: TaskOutput):
    """
    Link entities using a base transformer model.
    """
    pass
    # entity_text = inputs["entity_text"]
    # entity_linker = EntityLinkerBaseTransformer(entity_text)
    # entity_linker.link()
    # outputs["entity_link"] = entity_linker.entity_link