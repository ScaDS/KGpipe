#!/usr/bin/env python3
"""
Show command for KGbench CLI.

This module handles showing detailed information about components.
"""
import click
import yaml
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kgpipe.common.discovery import (
    discover_entry_points, find_task_by_name, find_pipeline_by_name
)
from kgpipe.evaluation.aspects.reference import ReferenceConfig
from kgpipe.evaluation.aspects.semantic import SemanticConfig
from kgpipe.evaluation.aspects.statistical import StatisticalConfig
from kgpipe.evaluation.util import get_metric_config_template

# Initialize Rich console for pretty output
console = Console()

def show_pipeline_details(pipeline_file: str):
    """Show details of a pipeline file."""
    try:
        with open(pipeline_file, 'r') as f:
            pipeline_data = yaml.safe_load(f)
        
        console.print(Panel(f"[bold blue]Pipeline Details:[/bold blue] {pipeline_file}"))
        
        # Show basic info
        table = Table()
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Name", pipeline_data.get('name', 'N/A'))
        table.add_row("Description", pipeline_data.get('description', 'N/A'))
        table.add_row("Version", pipeline_data.get('version', 'N/A'))
        
        console.print(table)
        
        # Show tasks
        if 'tasks' in pipeline_data:
            console.print("\n[bold]Tasks:[/bold]")
            task_table = Table()
            task_table.add_column("Task", style="cyan")
            task_table.add_column("Type", style="magenta")
            task_table.add_column("Input", style="yellow")
            task_table.add_column("Output", style="blue")
            
            for task in pipeline_data['tasks']:
                task_table.add_row(
                    task.get('name', 'N/A'),
                    task.get('type', 'N/A'),
                    str(task.get('input', 'N/A')),
                    str(task.get('output', 'N/A'))
                )
            
            console.print(task_table)
        
    except Exception as e:
        console.print(f"[red]Error reading pipeline file:[/red] {e}")


def show_run_details(run_file: str):
    """Show details of a run file."""
    try:
        with open(run_file, 'r') as f:
            run_data = json.load(f)
        
        console.print(Panel(f"[bold blue]Run Details:[/bold blue] {run_file}"))
        
        table = Table()
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Run ID", run_data.get('run_id', 'N/A'))
        table.add_row("Pipeline", run_data.get('pipeline', 'N/A'))
        table.add_row("Status", run_data.get('status', 'N/A'))
        table.add_row("Start Time", run_data.get('start_time', 'N/A'))
        table.add_row("End Time", run_data.get('end_time', 'N/A'))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error reading run file:[/red] {e}")


def show_report_details(report_file: str):
    """Show details of a report file."""
    try:
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        console.print(Panel(f"[bold blue]Report Details:[/bold blue] {report_file}"))
        
        table = Table()
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Report ID", report_data.get('report_id', 'N/A'))
        table.add_row("Run ID", report_data.get('run_id', 'N/A'))
        table.add_row("Generated", report_data.get('generated', 'N/A'))
        
        console.print(table)
        
        # Show metrics if available
        if 'metrics' in report_data:
            console.print("\n[bold]Metrics:[/bold]")
            metric_table = Table()
            metric_table.add_column("Metric", style="cyan")
            metric_table.add_column("Value", style="green")
            
            for metric_name, metric_value in report_data['metrics'].items():
                metric_table.add_row(metric_name, str(metric_value))
            
            console.print(metric_table)
        
    except Exception as e:
        console.print(f"[red]Error reading report file:[/red] {e}")


def show_task_details(task_name: str):
    """Show details of a task."""
    # Trigger discovery to ensure all tasks are registered
    discover_entry_points()
    
    task = find_task_by_name(task_name)
    if not task:
        console.print(f"[red]Task not found:[/red] {task_name}")
        return
    
    console.print(Panel(f"[bold blue]Task Details:[/bold blue] {task_name}"))
    
    table = Table()
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Name", getattr(task, 'name', 'N/A'))
    table.add_row("Category", getattr(task, 'category', 'N/A'))
    table.add_row("Description", getattr(task, 'description', 'N/A'))
    table.add_row("Input Spec", str(getattr(task, 'input_spec', 'N/A')))
    table.add_row("Output Spec", str(getattr(task, 'output_spec', 'N/A')))
    
    console.print(table)


def show_metric_config_templates():
    """Show YAML templates for metric config models."""
    templates = [
        ("ReferenceConfig", get_metric_config_template(ReferenceConfig)),
        ("StatisticalConfig", get_metric_config_template(StatisticalConfig)),
        ("SemanticConfig", get_metric_config_template(SemanticConfig)),
    ]

    for idx, (_, template) in enumerate(templates):
        if idx > 0:
            click.echo("---")
        click.echo(template.rstrip())


@click.group(name="show", invoke_without_command=True)
@click.argument("item", type=str, required=False)
@click.option(
    "--type", 
    "-t", 
    type=click.Choice(["pipeline", "run", "report", "task"]), 
    help="Type of item to show"
)
@click.pass_context
def show_cmd(ctx: click.Context, item: Optional[str], type: Optional[str]):
    """
    Show detailed information about an item.
    
    ITEM: Name or path of the item to show details for.
    """
    if ctx.invoked_subcommand:
        return

    if not item:
        console.print(ctx.get_help())
        return

    # Keep legacy `kgpipe show <item>` behavior while supporting
    # `kgpipe show metric-config-templates`.
    if item == "metric-config-templates":
        show_metric_config_templates()
        return

    # Auto-detect type if not specified
    if not type:
        if item.endswith('.yaml') or item.endswith('.yml'):
            type = "pipeline"
        elif item.endswith('.json'):
            # Try to determine if it's a run or report file
            try:
                with open(item, 'r') as f:
                    data = json.load(f)
                if 'run_id' in data and 'pipeline' in data:
                    type = "run"
                elif 'report_id' in data:
                    type = "report"
                else:
                    type = "run"  # Default to run
            except:
                type = "task"  # Default to task if file reading fails
        else:
            type = "task"  # Default to task
    
    if type == "pipeline":
        show_pipeline_details(item)
    elif type == "run":
        show_run_details(item)
    elif type == "report":
        show_report_details(item)
    elif type == "task":
        show_task_details(item)
    else:
        console.print(f"[red]Unknown type:[/red] {type}") 


@show_cmd.command(name="metric-config-templates")
def show_metric_config_templates_cmd():
    """Show YAML templates for evaluation metric configs."""
    show_metric_config_templates()