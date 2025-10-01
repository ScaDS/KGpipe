"""
Named Entity Recognition Tools

This module contains tools for named entity recognition.
"""

from kgpipe.common.registry import Registry

# Import tasks
from .corenlp_ner import corenlp_kbp_extraction_task

# Register tasks
def get_corenlp_kbp_extraction_task():
    return corenlp_kbp_extraction_task

Registry.register("task")(get_corenlp_kbp_extraction_task)

__all__ = [
    "corenlp_kbp_extraction_task"
] 