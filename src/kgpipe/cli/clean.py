#!/usr/bin/env python3
"""
Clean command for KGbench CLI.

This module handles cleaning up temporary files and cache.
"""

import shutil
import sys
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize Rich console for pretty output
console = Console()


def find_cache_files() -> List[Path]:
    """Find cache files to clean."""
    cache_dirs = [
        Path.cwd() / "kgpipe_cache",
        Path.cwd() / "kgpipe_output",
        Path.cwd() / "kgpipe_temp",
        Path.home() / ".kgpipe" / "cache"
    ]
    
    cache_files = []
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            cache_files.extend(cache_dir.rglob("*"))
    
    return cache_files


def find_temp_files() -> List[Path]:
    """Find temporary files to clean."""
    temp_dirs = [
        Path.cwd() / "kgpipe_temp",
        Path.cwd() / "tmp",
        Path("/tmp") / "kgpipe"
    ]
    
    temp_files = []
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            temp_files.extend(temp_dir.rglob("*"))
    
    return temp_files


@click.command()
@click.option(
    "--cache", 
    is_flag=True, 
    help="Remove cache files"
)
@click.option(
    "--temp", 
    is_flag=True, 
    help="Remove temporary files"
)
@click.option(
    "--all", 
    is_flag=True, 
    help="Remove all cache and temporary files"
)
@click.option(
    "--dry-run", 
    is_flag=True, 
    help="Show what would be removed without actually removing"
)
@click.pass_context
def clean_cmd(ctx: click.Context, cache: bool, temp: bool, all: bool, dry_run: bool):
    """
    Clean up temporary files and cache.
    
    Remove cache files, temporary files, or both to free up disk space.
    """
    if not any([cache, temp, all]):
        console.print("[yellow]No cleanup type specified. Use --cache, --temp, or --all[/yellow]")
        return
    
    files_to_remove = []
    
    if all or cache:
        cache_files = find_cache_files()
        files_to_remove.extend(cache_files)
        console.print(f"[dim]Found {len(cache_files)} cache files[/dim]")
    
    if all or temp:
        temp_files = find_temp_files()
        files_to_remove.extend(temp_files)
        console.print(f"[dim]Found {len(temp_files)} temporary files[/dim]")
    
    if not files_to_remove:
        console.print("[green]No files to clean up[/green]")
        return
    
    # Remove duplicates
    files_to_remove = list(set(files_to_remove))
    
    if dry_run:
        console.print(f"[yellow]DRY RUN - Would remove {len(files_to_remove)} files:[/yellow]")
        for file_path in files_to_remove[:10]:  # Show first 10
            console.print(f"  {file_path}")
        if len(files_to_remove) > 10:
            console.print(f"  ... and {len(files_to_remove) - 10} more")
        return
    
    # Confirm removal
    if not click.confirm(f"Remove {len(files_to_remove)} files?"):
        console.print("[yellow]Cleanup cancelled[/yellow]")
        return
    
    # Remove files
    removed_count = 0
    failed_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Cleaning up files...", total=len(files_to_remove))
        
        for file_path in files_to_remove:
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                removed_count += 1
            except Exception as e:
                failed_count += 1
                if ctx.obj["verbose"]:
                    console.print(f"[red]Failed to remove {file_path}:[/red] {e}")
            
            progress.update(task, advance=1)
    
    # Show results
    if failed_count == 0:
        console.print(f"[green]✓ Cleanup completed! Removed {removed_count} files[/green]")
    else:
        console.print(f"[yellow]⚠ Cleanup completed with errors. Removed {removed_count} files, failed {failed_count}[/yellow]") 