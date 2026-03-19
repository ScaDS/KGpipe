#!/usr/bin/env python3
"""
Test for dummy task discovery using kgpipe's module discovery.

This test demonstrates how to use kgpipe's discovery mechanism to discover
and test tasks from local modules.
"""

import sys
from pathlib import Path

import pytest

from kgpipe.common.discovery import (
    discover_local_modules,
    get_registered_tasks,
    find_task_by_name,
)
from kgpipe.common.registry import Registry
from kgpipe.common.models import DataFormat


# Add the src directory to Python path so modules can be discovered
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_discover_dummy_task_from_module():
    """Test discovering the dummy task using discover_local_modules."""
    # Verify the module file exists
    dummy_tasks_file = SRC_DIR / "dummy_tasks.py"
    assert dummy_tasks_file.exists(), f"Module file not found: {dummy_tasks_file}"
    
    # Use discover_local_modules to discover the module
    # Note: discover_local_modules converts the path to a module name.
    # Since SRC_DIR is in sys.path, we can discover modules from that directory.
    # We use the directory path - discover_local_modules will attempt to import
    # it as a module name (converted from the path).
    discover_local_modules(SRC_DIR)
    
    # Verify the task was discovered and registered using discovery methods
    tasks = get_registered_tasks()
    task_names = [task.name for task in tasks]
    
    assert "dummy_task" in task_names, f"dummy_task not found in registered tasks: {task_names}"
    
    # Get the task from registry using find_task_by_name from discovery module
    dummy_task = find_task_by_name("dummy_task")
    assert dummy_task is not None, "dummy_task should be found"
    
    # Verify task properties
    assert dummy_task.name == "dummy_task"
    assert "source" in dummy_task.input_spec
    assert dummy_task.input_spec["source"] == DataFormat.RDF_NTRIPLES
    assert "output" in dummy_task.output_spec
    assert dummy_task.output_spec["output"] == DataFormat.RDF_NTRIPLES


# def test_run_dummy_task():
#     """Test running the discovered dummy task."""
#     # Import to trigger registration
#     import dummy_tasks
    
#     # Get the task
#     dummy_task = Registry.get_task("dummy_task")
    
#     # Create temporary input and output files
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.nt', delete=False) as input_file:
#         input_file.write("<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> .\n")
#         input_path = Path(input_file.name)
    
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.nt', delete=False) as output_file:
#         output_path = Path(output_file.name)
    
#     try:
#         # Create Data objects
#         input_data = Data(path=input_path, format=DataFormat.RDF_NTRIPLES)
#         output_data = Data(path=output_path, format=DataFormat.RDF_NTRIPLES)
        
#         # Run the task
#         report = dummy_task.run(
#             inputs={"source": input_data},
#             outputs={"output": output_data}
#         )
        
#         # Verify the task completed successfully
#         assert report.status == "success", f"Task should succeed, but got: {report.status}"
        
#         # Verify output file was created
#         assert output_path.exists(), "Output file should be created"
        
#     finally:
#         # Cleanup
#         if input_path.exists():
#             input_path.unlink()
#         if output_path.exists():
#             output_path.unlink()


def test_discover_from_directory():
    """Test discovering tasks from a directory containing modules."""
    # Discover from the src directory using discover_local_modules
    discover_local_modules(SRC_DIR)
    
    # Verify the task was discovered using discovery methods
    tasks = get_registered_tasks()
    task_names = [task.name for task in tasks]
    
    assert "dummy_task" in task_names, f"dummy_task not found in registered tasks: {task_names}"
    
    # Use find_task_by_name from discovery module
    dummy_task = find_task_by_name("dummy_task")
    assert dummy_task is not None, "dummy_task should be found via find_task_by_name"
    assert dummy_task.name == "dummy_task"

