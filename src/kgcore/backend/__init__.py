# kgcore/backend/__init__.py
from .factory import create_backend, register_backend, get_available_backends, BackendFactory
from .memory import InMemoryGraph
from .rdf_file import RDFFileGraph

__all__ = [
    "create_backend",
    "register_backend", 
    "get_available_backends",
    "BackendFactory",
    "InMemoryGraph",
    "RDFFileGraph"
]

