"""
Entry-point discovery for KGbench.

This module provides automatic discovery of tasks, pipelines, and other components
from installed packages and local modules.
"""

import importlib
import importlib.util
import sys
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
        import kgpipe_llm.tasks
        
        print("Successfully discovered kgpipe_llm components")
        logger.info("Successfully discovered kgpipe_llm components")
        
    except ImportError as e:
        print(f"kgpipe_llm not available: {e}")
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


def _resolve_import_root(module_path: Path) -> tuple[Path, str] | None:
    """Return (sys.path root, dotted module prefix) for the longest matching sys.path entry."""
    module_path = module_path.resolve()
    best_match: tuple[Path, str] | None = None
    best_len = -1

    for sys_path_entry in sys.path:
        if not sys_path_entry:
            continue
        try:
            sys_path = Path(sys_path_entry).resolve()
            relative = module_path.relative_to(sys_path)
            if len(sys_path.parts) > best_len:
                best_match = (sys_path, ".".join(relative.parts))
                best_len = len(sys_path.parts)
        except (ValueError, OSError):
            continue

    return best_match


def _find_package_source_root(module_path: Path) -> Path | None:
    """Find the directory that should be on sys.path for package imports."""
    module_path = module_path.resolve()
    if module_path.is_file():
        module_path = module_path.parent

    for parent in [module_path, *module_path.parents]:
        try:
            relative = module_path.relative_to(parent)
        except ValueError:
            break
        if not relative.parts:
            continue

        top_package = parent / relative.parts[0]
        if top_package.is_dir() and (top_package / "__init__.py").exists():
            if not (parent / "__init__.py").exists():
                return parent

    return None


def _module_name_for_path(py_file: Path, scan_root: Path) -> str:
    """Build a dotted module name for a Python file under scan_root."""
    py_file = py_file.resolve()
    scan_root = scan_root.resolve()

    import_root = _resolve_import_root(scan_root)
    if import_root:
        sys_path_root, _ = import_root
        relative = py_file.relative_to(sys_path_root)
        return ".".join(relative.with_suffix("").parts)

    package_src = _find_package_source_root(scan_root)
    if package_src:
        path_str = str(package_src)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        relative = py_file.relative_to(package_src)
        return ".".join(relative.with_suffix("").parts)

    relative = py_file.relative_to(scan_root)
    return ".".join(relative.with_suffix("").parts)


def _import_python_module(py_file: Path, module_name: str) -> None:
    """Import a module by name, falling back to loading directly from a file path."""
    try:
        importlib.import_module(module_name)
        logger.info(f"Successfully discovered module: {module_name}")
        return
    except Exception as e:
        logger.debug(f"Could not import {module_name}: {e}")

    try:
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault(module_name, module)
            spec.loader.exec_module(module)
            logger.info(f"Successfully discovered module from file: {py_file} ({module_name})")
    except Exception as e:
        logger.warning(f"Error discovering module {py_file} ({module_name}): {e}")


def discover_local_modules(module_path: Path) -> None:
    """
    Discover components from local modules.
    
    This function can handle:
    - Python files (.py) - imports the file as a module
    - Directories - recursively scans for Python files and imports them
    - Paths in sys.path - converts to relative module names
    
    Args:
        module_path: Path to the module file or directory to scan
    """
    if not module_path.exists():
        logger.warning(f"Module path does not exist: {module_path}")
        return

    module_path = module_path.resolve()

    if module_path.is_file():
        if module_path.suffix != ".py" or module_path.name == "__init__.py":
            return
        py_files = [module_path]
        scan_root = module_path.parent
    elif module_path.is_dir():
        py_files = sorted(
            py_file
            for py_file in module_path.rglob("*.py")
            if py_file.name != "__init__.py"
        )
        scan_root = module_path
    else:
        return

    for py_file in py_files:
        module_name = _module_name_for_path(py_file, scan_root)
        _import_python_module(py_file, module_name)


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