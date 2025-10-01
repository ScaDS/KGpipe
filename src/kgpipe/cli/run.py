#!/usr/bin/env python3
"""
Run command for KGbench CLI.

This module handles pipeline execution commands.
"""

import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from kgpipe.common.models import KgPipe, Data, DataFormat
from kgpipe.execution.runner import PipelineRunner

# Placeholder classes for missing components
class YamlPipelineLoader:
    """Placeholder for YAML pipeline loader."""
    def load(self, pipeline_file: str) -> KgPipe:
        """Load pipeline from YAML file."""
        # For now, return a simple pipeline
        from kgpipe_tasks.matcher import paris_entity_matching_task
        return KgPipe(
            tasks=[paris_entity_matching_task],
            seed=Data(Path("placeholder.nt"), DataFormat.RDF)
        )

# Initialize Rich console for pretty output
console = Console()


def show_pipeline_plan(pipeline: KgPipe):
    """Display pipeline execution plan."""
    table = Table(title="Pipeline Execution Plan")
    table.add_column("Task", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Input", style="green")
    table.add_column("Output", style="yellow")
    
    for i, task in enumerate(pipeline.tasks, 1):
        table.add_row(
            f"{i}. {task.name}",
            task.__class__.__name__,
            str(task.input_spec) if hasattr(task, 'input_spec') else "N/A",
            str(task.output_spec) if hasattr(task, 'output_spec') else "N/A"
        )
    
    console.print(table)


def show_execution_summary(reports):
    """Display execution summary."""
    table = Table(title="Execution Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Output", style="blue")
    
    for report in reports:
        status = "✓ Success" if report.get('success', False) else "✗ Failed"
        duration = f"{report.get('duration', 0):.2f}s"
        output = str(report.get('output', 'N/A'))
        
        table.add_row(
            report.get('task_name', 'Unknown'),
            status,
            duration,
            output
        )
    
    console.print(table)


@click.command()
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option(
    "--output", 
    "-o", 
    type=click.Path(), 
    help="Output directory for results"
)
@click.option(
    "--temp-dir", 
    "-t", 
    type=click.Path(), 
    help="Temporary directory for intermediate files"
)
@click.option(
    "--dry-run", 
    is_flag=True, 
    help="Show execution plan without running"
)
@click.pass_context
def run_cmd(ctx: click.Context, pipeline_file: str, output: Optional[str], temp_dir: Optional[str], dry_run: bool):
    """
    Execute one pipeline file.
    
    PIPELINE_FILE: Path to the pipeline YAML file to execute
    """
    console.print(f"[bold blue]Executing pipeline:[/bold blue] {pipeline_file}")
    
    try:
        # Load pipeline
        loader = YamlPipelineLoader()
        pipeline = loader.load(pipeline_file)
        
        if dry_run:
            console.print("[yellow]DRY RUN - Showing execution plan:[/yellow]")
            show_pipeline_plan(pipeline)
            return
        
        # Set up output directory
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path.cwd() / "kgpipe_output"
            output_path.mkdir(exist_ok=True)
        
        # Execute pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing pipeline...", total=None)
            
            runner = PipelineRunner()
            # Create mock input data for now
            mock_input = Data(Path("mock_input.nt"), DataFormat.RDF)
            execution_report = runner.run_pipeline(pipeline, [mock_input])
            reports = [report.__dict__ for report in execution_report.task_reports]
            
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"[green]✓ Pipeline completed successfully![/green]")
        console.print(f"[dim]Results saved to:[/dim] {output_path}")
        
        # Show summary
        show_execution_summary(reports)
        
    except Exception as e:
        console.print(f"[red]✗ Pipeline execution failed:[/red] {e}")
        if ctx.obj["verbose"]:
            console.print_exception()
        sys.exit(1)


@click.command()
@click.argument("batch_file", type=click.Path(exists=True))
@click.option(
    "--output", 
    "-o", 
    type=click.Path(), 
    help="Output directory for results"
)
@click.option(
    "--parallel", 
    "-p", 
    type=int, 
    default=1, 
    help="Number of parallel executions"
)
@click.option(
    "--continue-on-error", 
    is_flag=True, 
    help="Continue execution even if some pipelines fail"
)
@click.pass_context
def batch_cmd(ctx: click.Context, batch_file: str, output: Optional[str], parallel: int, continue_on_error: bool):
    """
    Execute multiple pipelines from a batch file.
    
    BATCH_FILE: Path to the batch YAML file containing pipeline definitions
    """
    console.print(f"[bold blue]Executing batch:[/bold blue] {batch_file}")
    
    try:
        # Load batch configuration
        with open(batch_file, 'r') as f:
            batch_config = yaml.safe_load(f)
        
        pipelines = batch_config.get('pipelines', [])
        console.print(f"[dim]Found {len(pipelines)} pipelines to execute[/dim]")
        
        # Set up output directory
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path.cwd() / "kgpipe_batch_output"
            output_path.mkdir(exist_ok=True)
        
        # Execute pipelines
        successful = 0
        failed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing batch...", total=len(pipelines))
            
            for i, pipeline_config in enumerate(pipelines):
                try:
                    progress.update(task, description=f"Executing pipeline {i+1}/{len(pipelines)}")
                    
                    # Execute individual pipeline
                    pipeline_file = pipeline_config['file']
                    loader = YamlPipelineLoader()
                    pipeline = loader.load(pipeline_file)
                    
                    runner = PipelineRunner()
                    mock_input = Data(Path("mock_input.nt"), DataFormat.RDF)
                    execution_report = runner.run_pipeline(pipeline, [mock_input])
                    
                    successful += 1
                    
                except Exception as e:
                    failed += 1
                    console.print(f"[red]Pipeline {i+1} failed:[/red] {e}")
                    
                    if not continue_on_error:
                        raise e
                
                progress.update(task, advance=1)
        
        # Display results
        console.print(f"[green]✓ Batch completed![/green]")
        console.print(f"[dim]Successful:[/dim] {successful}, [dim]Failed:[/dim] {failed}")
        console.print(f"[dim]Results saved to:[/dim] {output_path}")
        
    except Exception as e:
        console.print(f"[red]✗ Batch execution failed:[/red] {e}")
        if ctx.obj["verbose"]:
            console.print_exception()
        sys.exit(1) 