#!/usr/bin/env python3
"""
Discover command for KGbench CLI.

This module handles discovery and registration of tasks from packages and modules.
"""

import importlib
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from kgpipe.common.discovery import (
    discover_entry_points,
    discover_local_modules,
    get_registered_tasks,
)

# Initialize Rich console for pretty output
console = Console()


def _function_path(func) -> str:
    return f"{func.__module__}.{func.__qualname__}"


def _component_name(factory) -> str:
    try:
        obj = factory()
        return getattr(obj, "name", factory.__name__)
    except Exception:
        return factory.__name__


def _show_component_table(title: str, items: list[tuple[str, str]]) -> None:
    if not items:
        return

    table = Table(title=title)
    table.add_column("Name", style="cyan")
    table.add_column("Function Path", style="green")

    for name, path in sorted(items, key=lambda item: item[0].lower()):
        table.add_row(name, path)

    console.print(table)
    console.print()


def _show_discovered_components() -> None:
    """Display discovered components with names and function paths."""
    from kgpipe.common.registry import Registry

    tasks = get_registered_tasks()
    pipelines = Registry.list("pipeline")
    metrics = Registry.list("metric")
    evaluators = Registry.list("evaluator")

    console.print()
    console.print("[bold blue]Discovered Components:[/bold blue]")
    console.print(
        f"Tasks: {len(tasks)}, Pipelines: {len(pipelines)}, "
        f"Metrics: {len(metrics)}, Evaluators: {len(evaluators)}"
    )
    console.print()

    _show_component_table(
        "Tasks",
        [(task.name, _function_path(task.function)) for task in tasks],
    )
    _show_component_table(
        "Pipelines",
        [(_component_name(factory), _function_path(factory)) for factory in pipelines],
    )
    _show_component_table(
        "Metrics",
        [(_component_name(factory), _function_path(factory)) for factory in metrics],
    )
    _show_component_table(
        "Evaluators",
        [(_component_name(factory), _function_path(factory)) for factory in evaluators],
    )


def discover_package(package_name: str) -> bool:
    """
    Discover and register components from a package by name.
    
    Args:
        package_name: Name of the package to import (e.g., 'kgpipe_tasks')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        importlib.import_module(package_name)
        console.print(f"[green]✓[/green] Successfully discovered package: {package_name}")
        return True
    except ImportError as e:
        console.print(f"[red]✗[/red] Failed to import package {package_name}: {e}")
        return False
    except Exception as e:
        console.print(f"[red]✗[/red] Error discovering package {package_name}: {e}")
        return False


def discover_module_path(module_path: str) -> bool:
    """
    Discover and register components from a module path.
    
    Args:
        module_path: Path to the module directory or file
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(module_path)
    if not path.exists():
        console.print(f"[red]✗[/red] Path does not exist: {module_path}")
        return False
    
    try:
        discover_local_modules(path)
        console.print(f"[green]✓[/green] Successfully discovered module path: {module_path}")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Error discovering module path {module_path}: {e}")
        return False


@click.command()
@click.option(
    "--package",
    "-p",
    "packages",
    multiple=True,
    help="Package name(s) to discover (e.g., 'kgpipe_tasks', 'kgpipe_llm')",
)
@click.option(
    "--module-path",
    "-m",
    "module_paths",
    multiple=True,
    type=click.Path(exists=True),
    help="Path(s) to module directory or file to discover (searches recursively)",
)
@click.option(
    "--all",
    "discover_all",
    is_flag=True,
    help="Discover from all default entry points (kgpipe_tasks, kgpipe_llm, etc.)",
)
@click.option(
    "--show-results",
    "-s",
    is_flag=True,
    help="Show discovered components after discovery",
)
@click.pass_context
def discover_cmd(
    ctx: click.Context,
    packages: tuple,
    module_paths: tuple,
    discover_all: bool,
    show_results: bool,
):
    """
    Discover and register tasks from packages or modules.
    
    This command imports packages or modules to trigger automatic registration
    of tasks, pipelines, metrics, and evaluators.
    
    Examples:
    
        # Discover from default entry points
        kgpipe discover --all
        
        # Discover from specific packages
        kgpipe discover --package kgpipe_tasks --package my_custom_tasks
        
        # Discover from module paths
        kgpipe discover --module-path ./my_tasks --module-path ./other_tasks
        
        # Combine options
        kgpipe discover --package kgpipe_tasks --module-path ./local_tasks --show-results
    """
    success_count = 0
    total_count = 0
    
    # Discover from default entry points if requested
    if discover_all:
        console.print("[bold blue]Discovering from default entry points...[/bold blue]")
        try:
            discover_entry_points()
            console.print("[green]✓[/green] Default entry points discovery completed")
            success_count += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Error during default discovery: {e}")
        total_count += 1
    
    # Discover from specified packages
    if packages:
        console.print(f"[bold blue]Discovering from {len(packages)} package(s)...[/bold blue]")
        for package in packages:
            total_count += 1
            if discover_package(package):
                success_count += 1
    
    # Discover from specified module paths
    if module_paths:
        console.print(f"[bold blue]Discovering from {len(module_paths)} module path(s)...[/bold blue]")
        for module_path in module_paths:
            total_count += 1
            if discover_module_path(module_path):
                success_count += 1
    
    # If no options specified, show help
    if not discover_all and not packages and not module_paths:
        console.print("[yellow]No discovery sources specified.[/yellow]")
        console.print("Use --all, --package, or --module-path to specify what to discover.")
        console.print("Run 'kgpipe discover --help' for more information.")
        return
    
    # Show summary
    console.print()
    console.print(f"[bold]Discovery Summary:[/bold] {success_count}/{total_count} successful")
    
    # Show discovered components if requested
    if show_results:
        _show_discovered_components()

