#!/usr/bin/env python3
"""
List command for KGbench CLI.

This module handles listing available components.
"""

import json
import csv
import sys
import click
from rich.console import Console
from rich.table import Table

from kgpipe.common.discovery import (
    discover_entry_points, get_registered_tasks, get_registered_pipelines,
    get_registered_metrics, get_registered_evaluators
)
from kgpipe.common.registry import Registry

# Initialize Rich console for pretty output
console = Console()

def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__

def function_location(func):
    return f"{func.__module__}.{func.__qualname__}"

def show_registered_tasks(format: str = "table") -> None:
    """Display registered tasks."""
    table = Table(title="Registered Tasks")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Description", style="green")
    table.add_column("Input", style="yellow")
    table.add_column("Output", style="blue")
    table.add_column("Full Name", style="red")
    

    tasks = get_registered_tasks()

    for task in tasks:
        table.add_row(
            task.name,
            ", ".join(getattr(task, 'category', [])),
            getattr(task, 'description', 'N/A'),
            str(getattr(task, 'input_spec', 'N/A')),
            str(getattr(task, 'output_spec', 'N/A')),
            "/".join(function_location(task.function).split(".")[:-1])
        )
    
    if format == "table":
        console.print(table)
    elif format == "csv":
        csv.writer(sys.stdout).writerows([[task.name, task.category, task.description, task.input_spec, task.output_spec, "/".join(function_location(task.function).split(".")[:-1])] for task in tasks])
    else:
        raise ValueError(f"Unknown format: {format}")
    print(f"Number of tasks: {len(tasks)}")


def show_registered_pipelines():
    """Display registered pipelines."""
    table = Table(title="Registered Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Tasks", style="yellow")
    table.add_column("Target", style="blue")
    
    pipelines = get_registered_pipelines()
    for pipeline in pipelines:
        table.add_row(
            getattr(pipeline, 'name', 'N/A'),
            getattr(pipeline, 'description', 'N/A'),
            str(len(getattr(pipeline, 'tasks', []))),
            str(getattr(pipeline, 'target', 'N/A'))
        )
    
    console.print(table)


def show_registered_metrics():
    """Display registered metrics."""
    table = Table(title="Registered Metrics")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Aspect", style="magenta")
    
    metrics = get_registered_metrics()
    for metric in metrics:
        aspect = getattr(metric, 'aspect', None)
        aspect_name = aspect.value if aspect else 'N/A'
        table.add_row(
            getattr(metric, 'name', 'N/A'),
            getattr(metric, 'description', 'N/A'),
            aspect_name
        )
    
    console.print(table)
    print(f"Number of metrics: {len(metrics)}")


def show_registered_evaluators():
    """Display registered evaluators."""
    table = Table(title="Registered Evaluators")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Supported Metrics", style="yellow")
    
    evaluators = get_registered_evaluators()
    for evaluator in evaluators:
        table.add_row(
            getattr(evaluator, 'name', 'N/A'),
            getattr(evaluator, 'description', 'N/A'),
            str(getattr(evaluator, 'supported_metrics', []))
        )
    
    console.print(table)


@click.command()
@click.option(
    "--type", 
    "-t", 
    type=click.Choice(["pipelines", "stages", "metrics", "tasks", "all"]), 
    default="all",
    help="Type of items to list"
)
@click.option(
    "--format", 
    "-f", 
    type=click.Choice(["table", "csv"]), 
    default="table",
    help="Output format"
)
@click.pass_context
def list_cmd(ctx: click.Context, type: str, format: str = "table"):
    """
    List available components.
    
    Lists tasks, pipelines, metrics, and evaluators that are available
    in the current KGbench installation.
    """
    # Trigger discovery to ensure all components are registered
    discover_entry_points()
    
    if type == "all" or type == "tasks":
        show_registered_tasks(format)
        console.print()  # Add spacing
    
    if type == "all" or type == "pipelines":
        show_registered_pipelines()
        console.print()  # Add spacing
    
    if type == "all" or type == "metrics":
        show_registered_metrics()
        console.print()  # Add spacing
    
    # if type == "all" or type == "stages":
    #     # Stages are similar to tasks, so we'll show tasks for now
    #     show_registered_tasks()
    #     console.print()  # Add spacing 