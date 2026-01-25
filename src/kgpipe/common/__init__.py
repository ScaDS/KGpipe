"""
Common utilities and models for KGbench.
"""

import logging
import logging.handlers

def setup_logging(log_file='app.log', level=logging.DEBUG):
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Check if the root logger already has handlers (avoid adding multiple)
    if not root_logger.handlers:
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add the handler to the root logger
        root_logger.addHandler(file_handler)

# Call this once at the start of your application
setup_logging()

from .models import (
    Data, DataFormat, KgTask, KgTaskReport, DynamicFormat, FormatRegistry,
    DataSet, KG, Metric, EvaluationReport, KgPipe, TaskInput, TaskOutput
)
from .registry import Registry
from .io import get_docker_volume_bindings, remap_data_path_for_container
from .discovery import (
    discover_entry_points, get_registered_tasks, get_registered_pipelines,
    get_registered_metrics, get_registered_evaluators, list_available_components,
    find_task_by_name, find_pipeline_by_name
)

__all__ = [
    "Data", "DataFormat", "KgTask", "KgTaskReport", "DynamicFormat", "FormatRegistry",
    "DataSet", "KG", "Stage", "Metric", "EvaluationReport", "KgPipe", "TaskInput", "TaskOutput",
    "Registry",
    "get_docker_volume_bindings", "remap_data_path_for_container",
    "discover_entry_points", "get_registered_tasks", "get_registered_pipelines",
    "get_registered_metrics", "get_registered_evaluators", "list_available_components",
    "find_task_by_name", "find_pipeline_by_name"
]
