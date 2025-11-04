# kgcore/model/__init__.py
from .base import KGGraph, KGEntity, KGRelation
from .graph import Triple, Node, Literal

__all__ = [
    "KGGraph",
    "KGEntity", 
    "KGRelation",
    "Triple",
    "Node",
    "Literal"
]

