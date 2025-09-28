#!/usr/bin/env python3
"""
Registry command for KGbench CLI.

This module handles registry management.
"""

import json
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table

from kgpipe.common.discovery import discover_entry_points
from kgpipe.common.registry import Registry

# Initialize Rich console for pretty output
console = Console()


def show_registry_table(registry_data):
    """Display registry data in a table."""
    table = Table(title="Registry Contents")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Items", style="yellow")
    
    for reg_type, items in registry_data.items():
        count = len(items)
        item_names = [item.get('name', str(item)) for item in items[:5]]  # Show first 5
        if count > 5:
            item_names.append(f"... and {count - 5} more")
        
        table.add_row(reg_type, str(count), ", ".join(item_names))
    
    console.print(table)


@click.command()
@click.option(
    "--type", 
    "-t", 
    type=str, 
    help="Show only specific registry type"
)
@click.option(
    "--format", 
    "-f", 
    type=click.Choice(["table", "json", "yaml"]), 
    default="table",
    help="Output format"
)
@click.pass_context
def registry_cmd(ctx: click.Context, type: Optional[str], format: str):
    """
    Show registry contents.
    
    Display information about registered tasks, pipelines, metrics, and evaluators.
    """
    # Trigger discovery to ensure all components are registered
    discover_entry_points()
    
    # Get registry data
    registry = Registry()
    
    if type:
        # Show specific type
        items = registry.list(type)
        registry_data = {type: items}
    else:
        # Show all types
        registry_data = {
            "tasks": registry.list("task"),
            "pipelines": registry.list("pipeline"),
            "metrics": registry.list("metric"),
            "evaluators": registry.list("evaluator")
        }
    
    # Display in requested format
    if format == "table":
        show_registry_table(registry_data)
    elif format == "json":
        console.print(json.dumps(registry_data, indent=2))
    elif format == "yaml":
        console.print(yaml.dump(registry_data, default_flow_style=False)) 