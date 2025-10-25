#!/usr/bin/env python3
"""
Task command for KGbench CLI.

This module handles task execution commands.
"""

import click
from typing import Optional, Tuple
from rich.console import Console
from typing import List
from kgpipe.common.models import Data, DataFormat
import re
from pathlib import Path
from kgpipe.common.registry import Registry
import sys
from kgpipe.common.models import KgTask

def parse_data_definition(data_definition: str) -> Tuple[Optional[str], Data]:
    """
    Parse a data definition string into a Data object.

    PATTERN: <path>|<format>@<short_name>

    short_name is optional and if not provided it will be the basename of the path
    """
    
    match = re.match(r"^(.*)\|([^@|]+)(?:@([^|]+))?$", data_definition)
    if not match:
        raise ValueError(f"Invalid data definition: {data_definition}")
    path = match.group(1)
    format = match.group(2)
    short_name = match.group(3) or None
    return short_name, Data(path=Path(path), format=DataFormat.from_extension(format), short_name=short_name)

console = Console()

@click.command()
@click.argument("task_name", type=str)
@click.option(
    "--input", 
    "-i", 
    type=str, 
    multiple=True,
    help="Inputs to the task"
)
@click.option(
    "--output", 
    "-o", 
    type=str, 
    multiple=True,
    help="Outputs of the task"
)
@click.option(
    "--config-file", 
    "-c", 
    type=click.Path(exists=True), 
    help="Path to the config file"
)
@click.pass_context
def task_cmd(ctx: click.Context, task_name: str, input: List[str], output: List[str], config_file: Optional[str]):
    """
    Execute a task.
    
    TASK_NAME: Name of the task to execute
    Inputs: Inputs to the task
    Outputs: Outputs of the task
    """

    inputs = [data for short_name, data in [parse_data_definition(i) for i in input]]
    outputs = [data for short_name, data in [parse_data_definition(o) for o in output]]

    console.print(f"[bold blue]Executing task:[/bold blue] {task_name}")
    console.print(f"[bold blue]Inputs:[/bold blue] {inputs}")
    console.print(f"[bold blue]Outputs:[/bold blue] {outputs}")
    
    def get_task(task_name: str) -> KgTask:
        try:
            return Registry.get_task(task_name)
        except Exception as e:
            console.print(f"[bold red]Task not found:[/bold red] {e}")
            # print available tasks
            console.print(f"[bold blue]Available tasks:[/bold blue] {'\n  - '.join([str(t) for t in Registry.list('task')])}")
            sys.exit(1)

    task = get_task(task_name)

    report = task.run(inputs, outputs)
    if report.status == "success":
        console.print(f"[bold green]Task completed successfully:[/bold green] {report}")
    elif report.status == "failed":
        console.print(f"[bold red]Task failed:[/bold red] {report.error}")
        sys.exit(1)
    elif report.status == "skipped":
        console.print(f"[bold yellow]Task skipped:[/bold yellow] {report}")
        sys.exit(1)
    console.print(f"[bold green]Task completed successfully:[/bold green] {report}")