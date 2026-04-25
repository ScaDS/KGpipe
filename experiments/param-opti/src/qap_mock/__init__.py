"""
Mock implementation of the experiments described in `Quality_Aware_Pipelines.pdf`.

This package is intentionally self-contained and does not depend on KGpipe.
It simulates:
- A small configuration space (implementations + parameters)
- A "true" end-to-end quality objective
- A correlated approximate quality estimator
- Search strategies (default, random, quality-aware)
"""

from .models import PipelineFamily, SearchMethod

__all__ = ["PipelineFamily", "SearchMethod"]

