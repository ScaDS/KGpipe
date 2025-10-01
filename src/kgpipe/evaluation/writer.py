from kgpipe.evaluation import EvaluationReport, MetricResult
import json
import csv
from io import StringIO
from typing import Literal, Callable, Tuple, Optional

def render_as_table(report: EvaluationReport, show_details: bool = True):
    """Render evaluation report as a table with multiline JSON details.
    
    Args:
        report: The evaluation report to render
        show_details: Whether to display detailed JSON information (default: True)
    """
    # print("=" * 80)
    # print("EVALUATION REPORT")
    # print("=" * 80)
    
    for aspect_result in report.aspect_results:
        print(f"\n{aspect_result.aspect.value.upper()} EVALUATION")
        print("-" * 50)
        print(f"Overall Score: {aspect_result.overall_score:.3f}")
        print()
        
        if not aspect_result.metrics:
            print("No metrics available")
            continue
            
        # Print metrics in a table format
        print(f"{'Metric':<20} {'Value':<15} {'Normalized':<12}")
        print("-" * 50)
        
        for metric in aspect_result.metrics:
            render_metric_as_table(metric, show_details)

def render_as_table_multi(reports: list[EvaluationReport], format: Literal["pretty", "csv"] = "pretty", 
                         show_details: bool = True, 
                         cell_render_func: Optional[Callable[[MetricResult], str]] = None):
    """Render multiple evaluation reports as a table where each row is a KG report.
    
    Args:
        reports: List of evaluation reports to render
        format: Output format - "pretty" for terminal display, "csv" for CSV format
        show_details: Whether to display detailed JSON information (default: True)
        cell_render_func: Function to render each metric cell. Takes MetricResult and returns string.
                         Default is lambda m: f"{m.value:.3f}/{m.normalized_score:.3f}"
    """
    if not reports:
        print("No reports to display")
        return
    
    # Default cell render function
    if cell_render_func is None:
        cell_render_func = lambda metric: f"{metric.value:.3f}/{metric.normalized_score:.3f}"
    
    if format == "pretty":
        render_as_table_multi_pretty(reports, show_details, cell_render_func)
    elif format == "csv":
        return render_as_table_multi_csv(reports, cell_render_func)
    else:
        raise ValueError(f"Unknown format: {format}")

def render_as_table_multi_pretty(reports: list[EvaluationReport], show_details: bool = True,
                                cell_render_func: Optional[Callable[[MetricResult], str]] = None):
    """Render multiple evaluation reports as a pretty-formatted table for terminal display."""
    if not reports:
        print("No reports to display")
        return
    
    # Default cell render function
    if cell_render_func is None:
        cell_render_func = lambda metric: f"{metric.value:.3f}/{metric.normalized_score:.3f}"
    
    # Collect all unique metric names across all reports
    all_metric_names = set()
    for report in reports:
        for aspect_result in report.aspect_results:
            for metric in aspect_result.metrics:
                all_metric_names.add(metric.name)
    
    # Sort metric names for consistent column ordering
    metric_names = sorted(list(all_metric_names))
    
    # Calculate column widths for better alignment
    kg_name_width = max(20, max(len(report.kg.name if hasattr(report, 'kg') and hasattr(report.kg, 'name') else f"KG_{i+1}") 
                                for i, report in enumerate(reports)))
    metric_widths = {}
    for metric_name in metric_names:
        # Calculate width needed for metric name + typical value format
        max_width = len(metric_name)
        for report in reports:
            for aspect_result in report.aspect_results:
                for metric in aspect_result.metrics:
                    if metric.name == metric_name:
                        value_str = cell_render_func(metric)
                        max_width = max(max_width, len(value_str))
        metric_widths[metric_name] = max_width + 2  # Add padding
    
    # Create header
    header = f"{'KG':<{kg_name_width}}"
    for metric_name in metric_names:
        header += f" {metric_name:<{metric_widths[metric_name]}}"
    print(header)
    print("-" * (kg_name_width + sum(metric_widths.values())))
    
    # Create rows for each report
    for i, report in enumerate(reports):
        kg_name = report.kg.name if hasattr(report, 'kg') and hasattr(report.kg, 'name') else f"KG_{i+1}"
        
        # Create a dictionary to store metric objects for this report
        report_metrics = {}
        
        # Collect all metrics from all aspects
        for aspect_result in report.aspect_results:
            for metric in aspect_result.metrics:
                report_metrics[metric.name] = metric
        
        # Build the row
        row = f"{kg_name:<{kg_name_width}}"
        for metric_name in metric_names:
            if metric_name in report_metrics:
                metric = report_metrics[metric_name]
                cell_content = cell_render_func(metric)
                row += f" {cell_content:<{metric_widths[metric_name]}}"
            else:
                row += f" {'N/A':<{metric_widths[metric_name]}}"
        
        print(row)
    
    print("-" * (kg_name_width + sum(metric_widths.values())))

def render_as_table_multi_csv(reports: list[EvaluationReport], 
                             cell_render_func: Optional[Callable[[MetricResult], str]] = None) -> str:
    """Render multiple evaluation reports as CSV format.
    
    Args:
        reports: List of evaluation reports to render
        cell_render_func: Function to render each metric cell. Takes MetricResult and returns string.
                         Default is lambda m: f"{m.value:.3f}/{m.normalized_score:.3f}"
    
    Returns:
        CSV string representation of the table
    """
    if not reports:
        return ""
    
    # Default cell render function
    if cell_render_func is None:
        cell_render_func = lambda metric: f"{metric.value:.3f}/{metric.normalized_score:.3f}"
    
    # Collect all unique metric names across all reports
    all_metric_names = set()
    for report in reports:
        for aspect_result in report.aspect_results:
            for metric in aspect_result.metrics:
                all_metric_names.add(metric.name)
    
    # Sort metric names for consistent column ordering
    metric_names = sorted(list(all_metric_names))
    
    # Create CSV output
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    header = ["KG"] + metric_names
    writer.writerow(header)
    
    # Write rows for each report
    for i, report in enumerate(reports):
        kg_name = report.kg.name if hasattr(report, 'kg') and hasattr(report.kg, 'name') else f"KG_{i+1}"
        
        # Create a dictionary to store metric objects for this report
        report_metrics = {}
        
        # Collect all metrics from all aspects
        for aspect_result in report.aspect_results:
            for metric in aspect_result.metrics:
                report_metrics[metric.name] = metric
        
        # Build the row
        row = [kg_name]
        for metric_name in metric_names:
            if metric_name in report_metrics:
                metric = report_metrics[metric_name]
                row.append(cell_render_func(metric))
            else:
                row.append("N/A")
        
        writer.writerow(row)
    
    return output.getvalue()

def render_metric_as_table(metric: MetricResult, show_details: bool = True):
    print(f"{metric.name:<20} {metric.value:<15.4f} {metric.normalized_score:<12.3f}")
    
    # Print details as pretty JSON (only if show_details is True)
    if show_details and metric.details:
        print("  Details:")
        try:
            # Pretty print JSON with proper indentation
            details_json = json.dumps(metric.details, indent=4, default=str, sort_keys=True)
            # Add indentation to each line for proper formatting
            for line in details_json.split('\n'):
                print(f"    {line}")
        except (TypeError, ValueError):
            # Fallback to string representation
            details_str = str(metric.details)
            print(f"    {details_str}")
        print()  # Empty line after details

