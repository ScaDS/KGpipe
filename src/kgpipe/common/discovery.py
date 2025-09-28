"""
Entry-point discovery for KGbench.

This module provides automatic discovery of tasks, pipelines, and other components
from installed packages and local modules.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging

from .registry import Registry
from .models import KgTask

logger = logging.getLogger(__name__)


def discover_entry_points() -> None:
    """
    Discover and register all entry points from installed packages.
    
    This function automatically discovers and registers:
    - Tasks from kgpipe_tasks and other packages
    - Pipelines from installed packages
    - Metrics from installed packages
    - Evaluators from installed packages
    """
    logger.info("Starting entry-point discovery...")
    
    # Discover from kgpipe_tasks
    discover_kgpipe_tasks()

    # Discover from kgpipe_llm
    discover_kgpipe_llm()
    
    # Discover evaluation components
    discover_evaluation_components()
    
    # Discover from other installed packages (future)
    discover_installed_packages()
    
    logger.info("Entry-point discovery completed")


def discover_kgpipe_tasks() -> None:
    """Discover and register components from kgpipe_tasks package."""
    try:
        import kgpipe_tasks
                
        logger.info("Successfully discovered kgpipe_tasks components")
        
    except ImportError as e:
        logger.warning(f"kgpipe_tasks not available: {e}")
        print(f"kgpipe_tasks not available: {e}")
    except Exception as e:
        logger.error(f"Error discovering kgpipe_tasks: {e}")
        print(f"Error discovering kgpipe_tasks: {e}")


def discover_kgpipe_llm() -> None:
    """Discover and register components from kgpipe_llm package."""
    try:
        import kgpipe_llm
        
        logger.info("Successfully discovered kgpipe_llm components")
        
    except ImportError as e:
        logger.warning(f"kgpipe_llm not available: {e}")
    except Exception as e:
        logger.error(f"Error discovering kgpipe_llm: {e}")

def discover_evaluation_components() -> None:
    """Discover and register evaluation components."""
    try:
        # Import metrics registration to trigger registration
        import kgpipe.evaluation.metrics
        
        logger.info("Successfully discovered evaluation components")
        
    except ImportError as e:
        logger.warning(f"Evaluation components not available: {e}")
    except Exception as e:
        logger.error(f"Error discovering evaluation components: {e}")


def discover_installed_packages() -> None:
    """Discover components from other installed packages (future implementation)."""
    # This is a placeholder for future package discovery
    # Could use pkg_resources or importlib.metadata to find entry points
    pass


def discover_local_modules(module_path: Path) -> None:
    """
    Discover components from local modules.
    
    Args:
        module_path: Path to the module directory to scan
    """
    if not module_path.exists():
        logger.warning(f"Module path does not exist: {module_path}")
        return
    
    try:
        # Convert path to module name
        module_name = str(module_path).replace('/', '.').replace('\\', '.')
        
        # Import the module
        module = importlib.import_module(module_name)
        
        # Look for registration functions
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and hasattr(attr, '_registry_kind'):
                # This is a registered function
                logger.info(f"Discovered registered function: {attr_name}")
                
    except ImportError as e:
        logger.warning(f"Could not import module {module_path}: {e}")
    except Exception as e:
        logger.error(f"Error discovering local modules: {e}")


def get_registered_tasks() -> List[Any]:
    """
    Get all registered tasks as task objects.
    
    Returns:
        List of KgTask objects
    """
    task_functions = Registry.list("task")
    tasks = []
    
    for task_function in task_functions:
        if isinstance(task_function, KgTask):
            tasks.append(task_function)
        else:
            logger.warning(f"Task {task_function} is not a KgTask object")
 
    return tasks


def get_registered_pipelines() -> List[Any]:
    """
    Get all registered pipelines.
    
    Returns:
        List of pipeline objects
    """
    pipeline_functions = Registry.list("pipeline")
    pipelines = []
    
    for pipeline_function in pipeline_functions:
        try:
            pipeline = pipeline_function()
            pipelines.append(pipeline)
        except Exception as e:
            logger.error(f"Error instantiating pipeline {pipeline_function.__name__}: {e}")
    
    return pipelines


def get_registered_metrics() -> List[Any]:
    """
    Get all registered metrics.
    
    Returns:
        List of metric objects
    """
    metric_functions = Registry.list("metric")
    metrics = []
    
    for metric_function in metric_functions:
        try:
            metric = metric_function()
            metrics.append(metric)
        except Exception as e:
            logger.error(f"Error instantiating metric {metric_function.__name__}: {e}")
    
    return metrics


def get_registered_evaluators() -> List[Any]:
    """
    Get all registered evaluators.
    
    Returns:
        List of evaluator objects
    """
    evaluator_functions = Registry.list("evaluator")
    evaluators = []
    
    for evaluator_function in evaluator_functions:
        try:
            evaluator = evaluator_function()
            evaluators.append(evaluator)
        except Exception as e:
            logger.error(f"Error instantiating evaluator {evaluator_function.__name__}: {e}")
    
    return evaluators


def list_available_components() -> Dict[str, List[Any]]:
    """
    List all available components by type.
    
    Returns:
        Dictionary mapping component types to lists of components
    """
    # Trigger discovery if not already done
    discover_entry_points()
    
    return {
        "tasks": get_registered_tasks(),
        "pipelines": get_registered_pipelines(),
        "metrics": get_registered_metrics(),
        "evaluators": get_registered_evaluators()
    }


def find_task_by_name(task_name: str) -> Optional[Any]:
    """
    Find a task by name.
    
    Args:
        task_name: Name of the task to find
        
    Returns:
        KgTask object if found, None otherwise
    """
    tasks = get_registered_tasks()
    return next((task for task in tasks if task.name == task_name), None)


def find_pipeline_by_name(pipeline_name: str) -> Optional[Any]:
    """
    Find a pipeline by name.
    
    Args:
        pipeline_name: Name of the pipeline to find
        
    Returns:
        Pipeline object if found, None otherwise
    """
    pipelines = get_registered_pipelines()
    return next((pipeline for pipeline in pipelines if pipeline.name == pipeline_name), None) 