# reimport all tasks from kgpipe_llm

from .alignment.json_alignment import json_ontology_mapping
from .construct.construct import (
    construct_from_any_input_as_triples,
    construct_from_any_input_with_kg_and_ontology_as_triples,
    construct_from_any_input_with_ontology_as_triples,
    construct_from_any_input_as_json_ld,
    construct_from_any_input_with_ontology_as_json_ld,
    construct_from_any_input_with_kg_as_json_ld,
    construct_from_any_input_with_kg_and_ontology_as_json_ld
)

from .text_triple_extract import llm_task_text_triple_extract_v1
from .rdf_om import llm_task_rdf_ontology_matching_v1, map_er_match_relations
from .json_mapping import llm_task_map_and_construct
__all__ = [
    "json_ontology_mapping", 
    "construct_from_any_input_as_triples", 
    "construct_from_any_input_with_kg_and_ontology_as_triples", 
    "construct_from_any_input_with_ontology_as_triples", 
    "construct_from_any_input_as_json_ld", 
    "construct_from_any_input_with_ontology_as_json_ld", 
    "construct_from_any_input_with_kg_as_json_ld", 
    "construct_from_any_input_with_kg_and_ontology_as_json_ld",
    "llm_task_text_triple_extract_v1",
    "llm_task_rdf_ontology_matching_v1",
    "map_er_match_relations",
    "llm_task_map_and_construct",
    # "aggregate_rdf_files"
]