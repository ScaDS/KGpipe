"""
Execution module for KGbench.

This module provides utilities for executing tasks and pipelines.
"""

from .docker import docker_client, run_docker_task
from .runner import PipelineRunner

__all__ = [
    "docker_client",
    "run_docker_task", 
    "PipelineRunner"
]
