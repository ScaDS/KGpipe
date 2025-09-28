#!/usr/bin/env python3
"""
Report command for KGbench CLI.

This module handles report generation.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

# Initialize Rich console for pretty output
console = Console()


def generate_markdown_report(evaluation_data):
    """Generate markdown report from evaluation data."""
    report = f"""# Evaluation Report

## Summary
- **Evaluation ID**: {evaluation_data.get('id', 'N/A')}
- **Ground Truth**: {evaluation_data.get('ground_truth', 'N/A')}
- **Prediction**: {evaluation_data.get('prediction', 'N/A')}

## Metrics
"""
    
    for metric_name, metric_value in evaluation_data.get('metrics', {}).items():
        report += f"- **{metric_name}**: {metric_value:.4f}\n"
    
    return report


def generate_html_report(evaluation_data):
    """Generate HTML report from evaluation data."""
    report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ margin: 10px 0; }}
        .value {{ font-weight: bold; color: #0066cc; }}
    </style>
</head>
<body>
    <h1>Evaluation Report</h1>
    <h2>Summary</h2>
    <p><strong>Evaluation ID:</strong> {evaluation_data.get('id', 'N/A')}</p>
    <p><strong>Ground Truth:</strong> {evaluation_data.get('ground_truth', 'N/A')}</p>
    <p><strong>Prediction:</strong> {evaluation_data.get('prediction', 'N/A')}</p>
    
    <h2>Metrics</h2>
"""
    
    for metric_name, metric_value in evaluation_data.get('metrics', {}).items():
        report += f'    <div class="metric"><strong>{metric_name}:</strong> <span class="value">{metric_value:.4f}</span></div>\n'
    
    report += """</body>
</html>"""
    
    return report


def generate_pdf_report(evaluation_data):
    """Generate PDF report from evaluation data."""
    # For now, return a simple text representation
    # In a real implementation, you'd use a library like reportlab or weasyprint
    return f"PDF Report for {evaluation_data.get('id', 'N/A')}"


@click.command()
@click.argument("evaluation_file", type=click.Path(exists=True))
@click.option(
    "--format", 
    "-f", 
    type=click.Choice(["markdown", "html", "pdf"]), 
    default="markdown",
    help="Output format for report"
)
@click.option(
    "--output", 
    "-o", 
    type=click.Path(), 
    help="Output file for report"
)
@click.pass_context
def report_cmd(ctx: click.Context, evaluation_file: str, format: str, output: Optional[str]):
    """
    Generate a report from evaluation results.
    
    EVALUATION_FILE: Path to the evaluation results file
    """
    console.print(f"[bold blue]Generating {format} report from:[/bold blue] {evaluation_file}")
    
    try:
        # Load evaluation data
        with open(evaluation_file, 'r') as f:
            evaluation_data = json.load(f)
        
        # Generate report
        if format == "markdown":
            report_content = generate_markdown_report(evaluation_data)
            extension = ".md"
        elif format == "html":
            report_content = generate_html_report(evaluation_data)
            extension = ".html"
        elif format == "pdf":
            report_content = generate_pdf_report(evaluation_data)
            extension = ".pdf"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Determine output file
        if output:
            output_file = Path(output)
        else:
            input_path = Path(evaluation_file)
            output_file = input_path.with_suffix(extension)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        console.print(f"[green]✓ Report generated successfully![/green]")
        console.print(f"[dim]Report saved to:[/dim] {output_file}")
        
    except Exception as e:
        console.print(f"[red]✗ Report generation failed:[/red] {e}")
        if ctx.obj["verbose"]:
            console.print_exception()
        sys.exit(1) 