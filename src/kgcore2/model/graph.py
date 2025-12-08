# kgcore/model/graph.py
"""
Compatibility module for kgback package.
Provides Triple, Node, and Literal classes that kgback expects.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from kgcore.common.types import KGId, Lit as KGLit

@dataclass
class Node:
    """Node class for compatibility with kgback."""
    id: KGId
    labels: list[str] = None
    props: dict[str, Any] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.props is None:
            self.props = {}

@dataclass  
class Triple:
    """Triple class for compatibility with kgback."""
    subject: KGId
    predicate: KGId
    object: KGId
    props: dict[str, Any] = None
    
    def __post_init__(self):
        if self.props is None:
            self.props = {}

class Literal:
    """Literal class for compatibility with kgback."""
    
    def __init__(self, value: str, datatype: Optional[str] = None):
        self.value = value
        self.datatype = datatype
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        if self.datatype:
            return f'Literal("{self.value}", datatype="{self.datatype}")'
        return f'Literal("{self.value}")'
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Literal):
            return self.value == other.value and self.datatype == other.datatype
        return False

