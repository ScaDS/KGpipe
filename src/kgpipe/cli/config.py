#!/usr/bin/env python3
"""
Config command for KGbench CLI.

This module handles configuration management.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table

# Initialize Rich console for pretty output
console = Console()


def get_default_config():
    """Get default configuration."""
    return {
        "output_dir": "kgpipe_output",
        "temp_dir": "kgpipe_temp",
        "log_level": "INFO",
        "parallel_jobs": 1,
        "docker_timeout": 3600,
        "cache_dir": "kgpipe_cache"
    }


def get_config_file():
    """Get the path to the configuration file."""
    config_dir = Path.home() / ".kgpipe"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "config.yaml"


def load_config():
    """Load configuration from file."""
    config_file = get_config_file()
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        return get_default_config()


def save_config(config):
    """Save configuration to file."""
    config_file = get_config_file()
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


@click.command()
@click.option(
    "--get", 
    "-g", 
    type=str, 
    help="Get configuration value"
)
@click.option(
    "--set", 
    "-s", 
    nargs=2, 
    type=str, 
    help="Set configuration value (key value)"
)
@click.option(
    "--edit", 
    "-e", 
    is_flag=True, 
    help="Edit configuration in default editor"
)
@click.pass_context
def config_cmd(ctx: click.Context, get: Optional[str], set: Optional[tuple], edit: bool):
    """
    Manage KGbench configuration.
    
    Get, set, or edit configuration values.
    """
    config = load_config()
    
    if get:
        # Get specific configuration value
        if get in config:
            console.print(f"[green]{get}:[/green] {config[get]}")
        else:
            console.print(f"[red]Configuration key not found:[/red] {get}")
            sys.exit(1)
    
    elif set:
        # Set configuration value
        key, value = set
        
        # Try to convert value to appropriate type
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        
        config[key] = value
        save_config(config)
        console.print(f"[green]✓ Set {key} = {value}[/green]")
    
    elif edit:
        # Edit configuration in default editor
        config_file = get_config_file()
        
        # Ensure config file exists
        if not config_file.exists():
            save_config(get_default_config())
        
        # Open in default editor
        editor = os.environ.get('EDITOR', 'nano')
        try:
            subprocess.run([editor, str(config_file)], check=True)
            console.print(f"[green]✓ Configuration edited in {editor}[/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗ Failed to open editor:[/red] {e}")
            sys.exit(1)
    
    else:
        # Show all configuration
        table = Table(title="KGbench Configuration")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Type", style="magenta")
        
        for key, value in config.items():
            table.add_row(key, str(value), type(value).__name__)
        
        console.print(table)
        console.print(f"[dim]Configuration file:[/dim] {get_config_file()}") 