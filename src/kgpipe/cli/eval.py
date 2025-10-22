#!/usr/bin/env python3
"""
Eval command for KGbench CLI.

This module handles evaluation commands.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from ..common.models import KG, DataFormat
from ..evaluation import Evaluator, EvaluationConfig, EvaluationAspect

# Initialize Rich console for pretty output
console = Console()


def show_evaluation_results(evaluation_report):
    """Display evaluation results."""
    # Overall score
    console.print(f"[bold green]Overall Score: {evaluation_report.overall_score:.3f}[/bold green]")
    console.print("")
    
    # Aspect results
    for aspect_result in evaluation_report.aspect_results:
        console.print(f"[bold blue]{aspect_result.aspect.value.title()} Evaluation: {aspect_result.overall_score:.3f}[/bold blue]")
        
        if aspect_result.metrics:
            table = Table(title=f"{aspect_result.aspect.value.title()} Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Normalized Score", style="yellow")
            table.add_column("Details", style="magenta")
            
            for metric in aspect_result.metrics:
                table.add_row(
                    metric.name,
                    f"{metric.value:.4f}",
                    f"{metric.normalized_score:.3f}",
                    f"{metric.details}"
                )
            
            console.print(table)
            console.print("")


def save_evaluation_results(evaluation_report, output_file: str):
    """Save evaluation results to file."""
    output_path = Path(output_file)
    
    if output_path.suffix.lower() == '.json':
        evaluation_report.to_json(output_path)
    elif output_path.suffix.lower() in ['.md', '.markdown']:
        evaluation_report.to_markdown(output_path)
    else:
        # Default to JSON
        evaluation_report.to_json(output_path)


@click.command()
@click.argument("target", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--ground-truth", 
    "-g", 
    type=click.Path(exists=True), 
    help="Path to ground truth KG"
)
@click.option(
    "--aspects", 
    "-a", 
    multiple=True, 
    type=click.Choice(['statistical', 'semantic', 'reference']),
    help="Evaluation aspects to compute (can specify multiple)"
)
@click.option(
    "--metrics", 
    "-m", 
    multiple=True, 
    help="Specific metrics to compute (can specify multiple)"
)
@click.option(
    "--output", 
    "-o", 
    type=click.Path(), 
    help="Output file for evaluation results"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(['json', 'markdown']),
    default='json',
    help="Output format for results"
)
@click.option( # flag
    "--debug",

)
@click.pass_context
def eval_cmd(ctx: click.Context, target: List[str], ground_truth: Optional[str], aspects: tuple, metrics: tuple, output: Optional[str], format: str, debug: Optional[str]):
    """
    Evaluate a knowledge graph against ground truth.
    
    TARGET: Path to the target knowledge graph to evaluate
    """
    console.print(f"[bold blue]Evaluating:[/bold blue] {target}")
    
    for tar in target:
        try:
            # Load target KG
            target_path = Path(tar)
            console.print(f"[dim]Loading target KG from:[/dim] {target_path}")
            
            # Determine format from file extension
            format_ext = target_path.suffix.lower().lstrip('.')
            try:
                kg_format = DataFormat(format_ext)
            except ValueError:
                kg_format = DataFormat.JSON  # Default to JSON
            
            target_kg = KG(
                id=str(target_path),
                name=target_path.stem,
                path=target_path,
                format=kg_format
            )
            
            # Load ground truth if provided
            reference_kg = None
            if ground_truth:
                ground_truth_path = Path(ground_truth)
                console.print(f"[dim]Loading ground truth from:[/dim] {ground_truth_path}")
                
                ref_format_ext = ground_truth_path.suffix.lower().lstrip('.')
                try:
                    ref_kg_format = DataFormat(ref_format_ext)
                except ValueError:
                    ref_kg_format = DataFormat.JSON
                
                reference_kg = KG(
                    id=str(ground_truth_path),
                    name=ground_truth_path.stem,
                    path=ground_truth_path,
                    format=ref_kg_format
                )
            
            # Set up evaluation configuration
            config = EvaluationConfig()
            
            # Set aspects if specified
            if aspects:
                config.aspects = [EvaluationAspect(aspect) for aspect in aspects]
            
            # Set metrics if specified
            if metrics:
                config.metrics = list(metrics)
            
            # Set output format
            config.output_format = format
            
            console.print(f"[dim]Evaluation aspects:[/dim] {', '.join([a.value for a in config.aspects])}")
            if config.metrics:
                console.print(f"[dim]Specific metrics:[/dim] {', '.join(config.metrics)}")
            
            # Run evaluation
            evaluator = Evaluator(config)
            evaluation_report = evaluator.evaluate(target_kg)
            
            # Display results
            console.print(f"[green]✓ Evaluation completed![/green]")
            show_evaluation_results(evaluation_report)
            
            # Save results if output file specified
            if output:
                save_evaluation_results(evaluation_report, output)
                console.print(f"[dim]Results saved to:[/dim] {output}")
            
        except Exception as e:
            console.print(f"[red]✗ Evaluation failed:[/red] {e}")
            if ctx.obj["verbose"]:
                console.print_exception()
            sys.exit(1) 
    
    if debug:
        from kgpipe.meta.systemgraph import SYS_KG
        # if has method asGraph, serialize it
        if hasattr(SYS_KG, "asGraph"):
            print(SYS_KG.asGraph().serialize(format="turtle"))
        else:
            print("SYS_KG does not have asGraph method")