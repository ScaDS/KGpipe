"""
Entity Linking Tools

This module contains tools for entity linking and disambiguation.
"""

from kgpipe.common.registry import Registry

# Import tasks
from .spotlight_entity_linking import dbpedia_spotlight_exchange, dbpedia_spotlight_ner_nel
from .falcon_entity_linking import falcon_ner_nel_rl, falcon_exchange
from .spacy_entity_linking import spacy_entity_linking_task
from .entity_linker import label_alias_embedding_el

# # Register tasks
# def get_spotlight_link_entities_task():
#     return spotlight_link_entities_task

# def get_spotlight_to_tejson_task():
#     return spotlight_to_tejson_task

# def get_falcon_link_entities_task():
#     return falcon_link_entities_task

# def get_falcon_to_tejson_task():
#     return falcon_to_tejson_task

# def get_spacy_entity_linking_task():
#     return spacy_entity_linking_task

# Registry.register("task")(get_spotlight_link_entities_task)
# Registry.register("task")(get_spotlight_to_tejson_task)
# Registry.register("task")(get_falcon_link_entities_task)
# Registry.register("task")(get_falcon_to_tejson_task)
# Registry.register("task")(get_spacy_entity_linking_task)

__all__ = [
    "dbpedia_spotlight_exchange",
    "dbpedia_spotlight_ner_nel",
    "falcon_ner_nel_rl",
    "falcon_exchange",
    "label_alias_embedding_el"
] 