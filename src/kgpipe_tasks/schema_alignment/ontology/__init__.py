from .ontology_rdf_filter import extract_ontology_from_rdf_task
from kgpipe.common.registry import Registry

def get_extract_ontology_from_rdf_task():
    return extract_ontology_from_rdf_task

Registry.register("task")(get_extract_ontology_from_rdf_task)

__all__ = [
    "extract_ontology_from_rdf_task"
]