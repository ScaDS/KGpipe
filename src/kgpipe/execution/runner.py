"""
Pipeline and Stage runners for executing KG pipelines.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.models import Data, DataFormat, KgPipe, KgTaskReport, Stage, TaskStatus


@dataclass
class ExecutionReport:
    """Report of a pipeline or stage execution."""
    
    pipeline_name: str
    start_time: float
    end_time: float
    status: TaskStatus
    task_reports: List[KgTaskReport] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get the total execution duration."""
        return self.end_time - self.start_time
    
    @property
    def success(self) -> bool:
        """Check if the execution was successful."""
        return self.status == TaskStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status.value,
            "success": self.success,
            "task_reports": [report.__dict__ for report in self.task_reports],
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class StageRunner:
    """Runner for executing individual pipeline stages."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def run_stage(self, stage: Stage, input_data: List[Data]) -> ExecutionReport:
        """Run a single stage with given input data."""
        start_time = time.time()
        
        self.logger.info(f"Starting stage: {stage.name}")
        
        try:
            # Validate inputs exist
            for data in input_data:
                if not data.exists():
                    raise FileNotFoundError(f"Input file not found: {data.path}")
            
            # Execute tasks in the stage
            task_reports = []
            for task in stage.tasks:
                self.logger.info(f"Running task: {task.name}")
                
                # For now, we'll create mock inputs/outputs
                # In a real implementation, you'd need to map stage inputs to task inputs
                mock_inputs = input_data[:1]  # Use first input for now
                mock_outputs = [Data(Path(f"output_{task.name}.ttl"), task.output_spec.get('output', DataFormat.RDF_TTL))]
                
                report = task.run(mock_inputs, mock_outputs)
                task_reports.append(report)
                
                if report.status == "failed":
                    raise RuntimeError(f"Task {task.name} failed")
            
            end_time = time.time()
            
            return ExecutionReport(
                pipeline_name=stage.name,
                start_time=start_time,
                end_time=end_time,
                status=TaskStatus.COMPLETED,
                task_reports=task_reports
            )
            
        except Exception as e:
            end_time = time.time()
            self.logger.error(f"Stage {stage.name} failed: {e}")
            
            return ExecutionReport(
                pipeline_name=stage.name,
                start_time=start_time,
                end_time=end_time,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )


class PipelineRunner:
    """Runner for executing complete KG pipelines."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.stage_runner = StageRunner(logger)
    
    def run_pipeline(self, pipeline: KgPipe, input_data: List[Data]) -> ExecutionReport:
        """Run a complete pipeline with given input data."""
        start_time = time.time()
        
        self.logger.info(f"Starting pipeline execution")
        
        try:
            # Validate pipeline
            if not pipeline.tasks:
                raise ValueError("Pipeline has no tasks to execute")
            
            # Validate inputs exist
            for data in input_data:
                if not data.exists():
                    raise FileNotFoundError(f"Input file not found: {data.path}")
            
            # Build execution plan
            execution_plan = pipeline.build(input_data[0])
            
            # Execute the pipeline
            task_reports = []
            for task_spec in execution_plan:
                task_name = task_spec["task"]
                task_inputs = task_spec["input"]
                task_outputs = task_spec["output"]
                
                # Find the corresponding task
                task = next((t for t in pipeline.tasks if t.name == task_name), None)
                if not task:
                    raise ValueError(f"Task {task_name} not found in pipeline")
                
                self.logger.info(f"Running task: {task_name}")
                
                # Validate task inputs exist
                for task_input in task_inputs:
                    if not task_input.exists():
                        raise FileNotFoundError(f"Task input file not found: {task_input.path}")
                
                # Run the task
                report = task.run(task_inputs, task_outputs)
                task_reports.append(report)
                
                if report.status == "failed":
                    raise RuntimeError(f"Task {task_name} failed")
                
                self.logger.info(f"Task {task_name} completed in {report.duration:.2f}s")
            
            end_time = time.time()
            
            return ExecutionReport(
                pipeline_name="pipeline",
                start_time=start_time,
                end_time=end_time,
                status=TaskStatus.COMPLETED,
                task_reports=task_reports
            )
            
        except Exception as e:
            end_time = time.time()
            self.logger.error(f"Pipeline execution failed: {e}")
            
            return ExecutionReport(
                pipeline_name="pipeline",
                start_time=start_time,
                end_time=end_time,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    def run_pipeline_with_stages(self, pipeline: KgPipe, input_data: List[Data]) -> ExecutionReport:
        """Run a pipeline by executing stages in dependency order."""
        start_time = time.time()
        
        self.logger.info(f"Starting pipeline execution with stages")
        
        try:
            # This would require the pipeline to have stages defined
            # For now, we'll fall back to the simple task-based execution
            return self.run_pipeline(pipeline, input_data)
            
        except Exception as e:
            end_time = time.time()
            self.logger.error(f"Pipeline execution failed: {e}")
            
            return ExecutionReport(
                pipeline_name="pipeline",
                start_time=start_time,
                end_time=end_time,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    def validate_pipeline(self, pipeline: KgPipe) -> List[str]:
        """Validate a pipeline before execution."""
        errors = []
        
        if not pipeline.tasks:
            errors.append("Pipeline has no tasks")
        
        # Check for duplicate task names
        task_names = [task.name for task in pipeline.tasks]
        if len(task_names) != len(set(task_names)):
            errors.append("Pipeline has duplicate task names")
        
        # Validate task specifications
        for task in pipeline.tasks:
            if not task.input_spec:
                errors.append(f"Task {task.name} has no input specification")
            if not task.output_spec:
                errors.append(f"Task {task.name} has no output specification")
        
        return errors 