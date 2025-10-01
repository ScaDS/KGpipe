"""
RDF-specific data integration tasks for Knowledge Graph construction.
"""

import json
from typing import Optional, List, Dict, Any
from .core import LLMClient, BaseTask, default_client
from .models import RDFTriple, RDFGraph, OntologyMappings

def extract_rdf_triples_from_text(text: str, namespace_prefixes: Optional[Dict[str, str]] = None,
                                 client: Optional[LLMClient] = None) -> Optional[RDFGraph]:
    """
    Extract RDF triples from text with proper URI formatting.
    
    Args:
        text: Input text to extract triples from
        namespace_prefixes: Dictionary of namespace prefixes (e.g., {"wd": "http://www.wikidata.org/entity/"})
        client: LLM client instance (uses default if None)
        
    Returns:
        RDFGraph object containing extracted triples or None if extraction fails
    """
    client = client or default_client
    
    # Prepare namespace context for the prompt
    namespace_context = ""
    if namespace_prefixes:
        namespace_context = f"\nUse these namespace prefixes: {json.dumps(namespace_prefixes)}"
    
    prompt = f"""Extract RDF triples from the following text.
    Format subjects and predicates as URIs using appropriate namespaces.
    Objects can be URIs or literals as appropriate.
    
    {namespace_context}
    
    Text: {text}
    
    Return the results as a JSON object with:
    - triples: array of RDF triples with subject, predicate, object
    - namespace_prefixes: the namespace prefixes used
    
    Each triple should have:
    - subject: URI of the subject
    - predicate: URI of the predicate  
    - object: URI or literal value of the object"""
    
    return client.send_prompt(prompt, RDFGraph)

def validate_rdf_triple(triple: RDFTriple, ontology_constraints: Optional[Dict[str, Any]] = None,
                       client: Optional[LLMClient] = None) -> bool:
    """
    Validate an RDF triple against ontology constraints.
    
    Args:
        triple: RDF triple to validate
        ontology_constraints: Constraints from the target ontology
        client: LLM client instance (uses default if None)
        
    Returns:
        True if triple is valid, False otherwise
    """
    client = client or default_client
    
    constraints_context = ""
    if ontology_constraints:
        constraints_context = f"\nOntology constraints: {json.dumps(ontology_constraints)}"
    
    prompt = f"""Validate this RDF triple against the ontology constraints.
    
    Triple: {triple.model_dump_json()}
    {constraints_context}
    
    Respond with true if the triple is valid according to the ontology constraints, false otherwise."""
    
    # For validation, we'll use a simple boolean response
    response = client.send_prompt(prompt, type(bool))
    return response if response is not None else False

def map_json_to_rdf_schema(json_data: str, target_ontology: str,
                          client: Optional[LLMClient] = None) -> Optional[OntologyMappings]:
    """
    Map JSON data structure to RDF schema/ontology.
    
    Args:
        json_data: JSON data or schema to map
        target_ontology: Target RDF ontology definition
        client: LLM client instance (uses default if None)
        
    Returns:
        OntologyMappings object or None if mapping fails
    """
    client = client or default_client
    
    prompt = f"""Map the JSON data structure to the target RDF ontology.
    
    JSON Data:
    {json_data}
    
    Target RDF Ontology:
    {target_ontology}
    
    Create mappings that specify:
    - source_field: field in the JSON data
    - target_class: RDF class in the ontology
    - target_property: RDF property in the ontology
    - mapping_type: type of mapping (direct, transform, etc.)
    - confidence: confidence score (0.0 to 1.0)
    
    Consider RDF-specific concepts like:
    - Classes vs Properties
    - Data properties vs Object properties
    - Namespace URIs
    - Literal vs URI values
    
    Respond with ontology mappings in JSON format."""
    
    return client.send_prompt(prompt, OntologyMappings)

def generate_rdf_queries(triples: List[RDFTriple], query_type: str = "sparql",
                        client: Optional[LLMClient] = None) -> Optional[str]:
    """
    Generate RDF queries from a set of triples.
    
    Args:
        triples: List of RDF triples
        query_type: Type of query to generate (sparql, graphql, etc.)
        client: LLM client instance (uses default if None)
        
    Returns:
        Generated query string or None if generation fails
    """
    client = client or default_client
    
    triples_json = json.dumps([triple.model_dump() for triple in triples])
    
    prompt = f"""Generate a {query_type.upper()} query based on these RDF triples.
    
    Triples: {triples_json}
    
    Generate a query that would retrieve or manipulate these triples.
    Return only the query string without additional explanation."""
    
    # For query generation, we'll return the raw string response
    response = client.send_prompt(prompt, "sparql")
    return response

class RDFExtractionTask(BaseTask[RDFGraph]):
    """Task for extracting RDF triples from text."""
    
    def execute(self, text: str, namespace_prefixes: Optional[Dict[str, str]] = None) -> Optional[RDFGraph]:
        """Execute RDF extraction task."""
        return extract_rdf_triples_from_text(text, namespace_prefixes, self.client)

class RDFValidationTask(BaseTask[bool]):
    """Task for validating RDF triples against ontology constraints."""
    
    def execute(self, triple: RDFTriple, ontology_constraints: Optional[Dict[str, Any]] = None) -> Optional[bool]:
        """Execute RDF validation task."""
        return validate_rdf_triple(triple, ontology_constraints, self.client)

class RDFMappingTask(BaseTask[OntologyMappings]):
    """Task for mapping JSON to RDF schema."""
    
    def execute(self, json_data: str, target_ontology: str) -> Optional[OntologyMappings]:
        """Execute RDF mapping task."""
        return map_json_to_rdf_schema(json_data, target_ontology, self.client)

# Utility functions for RDF processing
def merge_rdf_graphs(graphs: List[RDFGraph]) -> RDFGraph:
    """
    Merge multiple RDF graphs into a single graph.
    
    Args:
        graphs: List of RDF graphs to merge
        
    Returns:
        Merged RDF graph
    """
    all_triples = []
    all_namespaces = {}
    
    for graph in graphs:
        all_triples.extend(graph.triples)
        all_namespaces.update(graph.namespace_prefixes)
    
    return RDFGraph(triples=all_triples, namespace_prefixes=all_namespaces)

def filter_rdf_triples(triples: List[RDFTriple], subject_filter: Optional[str] = None,
                      predicate_filter: Optional[str] = None, object_filter: Optional[str] = None) -> List[RDFTriple]:
    """
    Filter RDF triples based on subject, predicate, or object patterns.
    
    Args:
        triples: List of RDF triples to filter
        subject_filter: Optional subject pattern to filter by
        predicate_filter: Optional predicate pattern to filter by
        object_filter: Optional object pattern to filter by
        
    Returns:
        Filtered list of RDF triples
    """
    filtered_triples = []
    
    for triple in triples:
        if subject_filter and subject_filter not in triple.subject:
            continue
        if predicate_filter and predicate_filter not in triple.predicate:
            continue
        if object_filter and object_filter not in triple.object:
            continue
        filtered_triples.append(triple)
    
    return filtered_triples 