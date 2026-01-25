#!/usr/bin/env python3
"""
Main CLI entry point for KGbench.

This module defines the main CLI group and imports all subcommands.
"""

from typing import Optional
import click
from rich.console import Console

# Import all subcommands
# from .run import run_cmd, batch_cmd
from .eval import eval_cmd
from .report import report_cmd
from .list import list_cmd
from .show import show_cmd
from .config import config_cmd
from .registry import registry_cmd
from .clean import clean_cmd
from .task import task_cmd
from .discover import discover_cmd

# Initialize Rich console for pretty output
console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="kgpipe")
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True), 
    help="Path to configuration file"
)
@click.option(
    "--verbose", 
    "-v", 
    is_flag=True, 
    help="Enable verbose output"
)
@click.option(
    "--quiet", 
    "-q", 
    is_flag=True, 
    help="Suppress output except errors"
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool, quiet: bool):
    """
    KGpipe - Knowledge Graph Pipeline Framework
    
    A comprehensive framework for creating, executing, and evaluating knowledge graph pipelines.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store global options in context
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    
    # Set up logging level
    if quiet:
        ctx.obj["log_level"] = "ERROR"
    elif verbose:
        ctx.obj["log_level"] = "DEBUG"
    else:
        ctx.obj["log_level"] = "INFO"


# Add all subcommands
# cli.add_command(run_cmd)
# cli.add_command(batch_cmd)
cli.add_command(eval_cmd)
cli.add_command(report_cmd)
cli.add_command(list_cmd)
cli.add_command(show_cmd)
cli.add_command(config_cmd)
cli.add_command(registry_cmd)
cli.add_command(clean_cmd)
cli.add_command(task_cmd)
cli.add_command(discover_cmd)


if __name__ == "__main__":
    cli() 