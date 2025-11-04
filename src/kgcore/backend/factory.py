# kgcore/backend/factory.py
from __future__ import annotations
from typing import Dict, Type, Any, Optional, Union
import importlib
from kgcore.model.base import KGGraph

class BackendFactory:
    """Factory for creating knowledge graph backends from various packages."""
    
    def __init__(self):
        self._backends: Dict[str, Dict[str, Any]] = {
            # kgcore built-in backends
            "memory": {
                "module": "kgcore.backend.memory",
                "class": "InMemoryGraph",
                "package": "kgcore"
            },
            "rdf_file": {
                "module": "kgcore.backend.rdf_file", 
                "class": "RDFFileGraph",
                "package": "kgcore"
            },
            # kgback backends
            "postgres": {
                "module": "kgback.postgres_back",
                "class": "PostgresECB", 
                "package": "kgback"
            },
            "sqlite": {
                "module": "kgback.sqlite_back",
                "class": "SqliteECB",
                "package": "kgback"
            },
            "sparql": {
                "module": "kgback.sparql_back",
                "class": "SparqlGraph",
                "package": "kgback"
            },
            "neo4j": {
                "module": "kgback.neo4j_back",
                "class": "Neo4jGraph", 
                "package": "kgback"
            }
        }
    
    def register_backend(self, name: str, module: str, class_name: str, package: str = None):
        """Register a new backend type."""
        self._backends[name] = {
            "module": module,
            "class": class_name,
            "package": package
        }
    
    def get_available_backends(self) -> list[str]:
        """Get list of available backend names."""
        return list(self._backends.keys())
    
    def create_backend(self, backend_name: str, **kwargs) -> KGGraph:
        """Create a backend instance by name."""
        if backend_name not in self._backends:
            available = ", ".join(self.get_available_backends())
            raise ValueError(f"Unknown backend: {backend_name}. Available backends: {available}")
        
        backend_info = self._backends[backend_name]
        
        try:
            # Try to import the module
            module = importlib.import_module(backend_info["module"])
            backend_class = getattr(module, backend_info["class"])
            
            # Create instance with provided kwargs
            return backend_class(**kwargs)
            
        except ImportError as e:
            package = backend_info.get("package", "unknown")
            raise ImportError(f"Could not import {backend_info['module']}.{backend_info['class']} "
                            f"from package '{package}'. Make sure '{package}' is installed. "
                            f"Original error: {e}")
        except AttributeError as e:
            raise AttributeError(f"Class {backend_info['class']} not found in module "
                               f"{backend_info['module']}. Original error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create backend '{backend_name}': {e}")

# Global factory instance
_factory = BackendFactory()

def create_backend(backend_name: str, **kwargs) -> KGGraph:
    """Convenience function to create a backend using the global factory."""
    return _factory.create_backend(backend_name, **kwargs)

def register_backend(name: str, module: str, class_name: str, package: str = None):
    """Convenience function to register a backend with the global factory."""
    _factory.register_backend(name, module, class_name, package)

def get_available_backends() -> list[str]:
    """Convenience function to get available backends from the global factory."""
    return _factory.get_available_backends()
