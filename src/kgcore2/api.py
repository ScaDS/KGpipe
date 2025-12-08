# kgcore/api.py
from __future__ import annotations
from typing import Literal, Optional
from kgcore.backend.factory import create_backend, get_available_backends
from kgcore.model.base import KGGraph

# Get available backends dynamically
_available_backends = get_available_backends()
BackendName = Literal[tuple(_available_backends)] if _available_backends else Literal["memory"]

def KG(backend: str = "memory", **kwargs) -> KGGraph:
    """
    Create a knowledge graph instance with the specified backend.
    
    Args:
        backend: Backend name (e.g., 'memory', 'rdf_file', 'postgres', 'sqlite', 'sparql')
        **kwargs: Backend-specific configuration parameters
        
    Returns:
        KGGraph instance
        
    Examples:
        # Memory backend (default)
        kg = KG()
        
        # RDF file backend
        kg = KG("rdf_file", file_path="my_kg.ttl", format="turtle")
        
        # PostgreSQL backend (requires kgback package)
        kg = KG("postgres", host="localhost", database="mydb", user="user", password="pass")
        
        # SQLite backend (requires kgback package)  
        kg = KG("sqlite", db_url="sqlite:///my_kg.db")
        
        # SPARQL backend (requires kgback package)
        kg = KG("sparql", endpoint="http://localhost:3030/ds/sparql")
    """
    return create_backend(backend, **kwargs)
