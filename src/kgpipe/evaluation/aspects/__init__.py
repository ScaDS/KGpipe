"""
Evaluation Aspects Package

Contains evaluators for the three main evaluation aspects.
"""

from .statistical import StatisticalEvaluator
from .semantic import SemanticEvaluator
from .reference import ReferenceEvaluator

__all__ = [
    "StatisticalEvaluator",
    "SemanticEvaluator", 
    "ReferenceEvaluator"
] 