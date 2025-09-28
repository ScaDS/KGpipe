"""
Evaluation Reports Module

Handles generation and export of evaluation reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from ..common.models import KG, Data
from .base import AspectResult, EvaluationConfig


@dataclass
class EvaluationReport:
    """Complete evaluation report containing all results."""
    kg: KG
    references: Dict[str, Data]
    aspect_results: List[AspectResult]
    overall_score: float
    config: EvaluationConfig
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError("Overall score must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "kg": {
                "name": self.kg.name,
                "path": str(self.kg.path),
                "format": self.kg.format.value
            },
            "reference_kg": {
                "name": self.reference_kg.name,
                "path": str(self.reference_kg.path),
                "format": self.reference_kg.format.value
            } if self.reference_kg else None,
            "overall_score": self.overall_score,
            "aspect_results": [
                {
                    "aspect": result.aspect.value,
                    "overall_score": result.overall_score,
                    "metrics": [
                        {
                            "name": metric.name,
                            "value": metric.value,
                            "normalized_score": metric.normalized_score,
                            "details": metric.details
                        }
                        for metric in result.metrics
                    ],
                    "details": result.details
                }
                for result in self.aspect_results
            ],
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def to_json(self, filepath: Optional[Path] = None) -> str:
        """Export report to JSON format."""
        report_dict = self.to_dict()
        json_str = json.dumps(report_dict, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def to_markdown(self, filepath: Optional[Path] = None) -> str:
        """Export report to Markdown format."""
        lines = [
            "# Knowledge Graph Evaluation Report",
            "",
            f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Target KG:** {self.kg.name}",
            ""
        ]
        
        if self.reference_kg:
            lines.extend([
                f"**Reference KG:** {self.reference_kg.name}",
                ""
            ])
        
        lines.extend([
            f"## Overall Score: {self.overall_score:.3f}",
            ""
        ])
        
        # Aspect results
        for result in self.aspect_results:
            lines.extend([
                f"## {result.aspect.value.title()} Evaluation",
                f"**Score:** {result.overall_score:.3f}",
                ""
            ])
            
            if result.metrics:
                lines.append("### Metrics")
                lines.append("")
                lines.append("| Metric | Value | Normalized Score |")
                lines.append("|--------|-------|------------------|")
                
                for metric in result.metrics:
                    lines.append(f"| {metric.name} | {metric.value:.4f} | {metric.normalized_score:.3f} |")
                lines.append("")
        
        markdown_content = "\n".join(lines)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(markdown_content)
        
        return markdown_content
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        summary = {
            "overall_score": self.overall_score,
            "aspect_scores": {
                result.aspect.value: result.overall_score
                for result in self.aspect_results
            },
            "total_metrics": sum(len(result.metrics) for result in self.aspect_results)
        }
        
        return summary