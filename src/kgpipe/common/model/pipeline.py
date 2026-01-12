import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union, Type
import json
from uuid import uuid4
import logging
import shutil
from rdflib import Graph
from pydantic import BaseModel, field_validator
from pydantic_core import core_schema

from .data import Data, DataFormat, DataSet
from .task import KgTask, KgTaskReport
# from .kg import KG

class KgPipePlanStep(BaseModel):
    """A step in a KG pipeline plan."""
    task: str
    input: List[Data]
    output: List[Data]

class KgPipePlan(BaseModel):
    """A KG pipeline plan."""
    steps: List[KgPipePlanStep]
    seed: Optional[Data] = None
    source: Optional[Data] = None
    result: Optional[Data] = None


    
    # def __str__(self) -> str:
    #     return f"KgTaskReport({self.task_name}, {self.status}, {self.duration:.2f}s)"

class KgStageReport(BaseModel):
    """Report of a stage execution."""
    stage_name: str
    task_reports: List[KgTaskReport]
    start_ts: float
    duration: float
    status: str
    error: Optional[str] = None


# @dataclass
# class Stage:
#     """Represents a stage in a pipeline, containing one or more tasks."""
#     id: str
#     name: str
#     description: Optional[str] = None
#     tasks: List[KgTask] = field(default_factory=list)
#     inputs: List[Union[DataSet, KG]] = field(default_factory=list)
#     outputs: List[Union[DataSet, KG]] = field(default_factory=list)
#     parameters: Dict[str, Any] = field(default_factory=dict)
#     depends_on: List[str] = field(default_factory=list)  # List of stage IDs this stage depends on

#     def __post_init__(self):
#         if not self.id:
#             self.id = str(uuid.uuid4())
#         if not self.name:
#             raise ValueError("Stage name cannot be empty")
    
#     def add_task(self, task: KgTask) -> None:
#         """Add a task to this stage."""
#         self.tasks.append(task)
    
#     def __str__(self) -> str:
#         return f"Stage({self.name}, tasks={len(self.tasks)})"




@dataclass
class KgPipe:
    """A KG pipeline using a list of tasks."""
    
    tasks: List[KgTask]
    seed: Data
    data_dir: str = ""
    data: List[Data] = field(default_factory=list)
    plan: KgPipePlan = field(default_factory=lambda: KgPipePlan(
        steps=[],
        seed=None,
        source=None,
        result=None
    ))

    def __post_init__(self):
        if not self.tasks:
            raise ValueError("Pipeline must have at least one task")
        if not self.seed:
            raise ValueError("Pipeline must have a target")
        
        # Initialize data list with target
        if not self.data:
            self.data = [self.seed]
    
    def add_data(self, data: Data) -> None:
        """Add additional data to the pipeline."""
        self.data.append(data)


    def build(self, source: Data, result: Optional[Data] = None, stable_files: bool = False) -> KgPipePlan:
        """Generate the execution plan as a list of dictionaries."""
        catalog = [source] + self.data
        calls: List[KgPipePlanStep] = []

        def gen_file_path(task: KgTask, format_spec: Format, prefix: str = "", suffix: str = ""):
            if stable_files:
                return Path(self.data_dir) / f"{prefix}{task.name}{suffix}.{format_spec.extension}"
            else:
                return Path(self.data_dir) / f"{prefix}{task.name}.{uuid4().hex}.{format_spec.extension}"
                

        for idx, task in enumerate(self.tasks):
            # Match inputs
            inputs = []
            for input_name, format_spec in task.input_spec.items():
                if input_name == "kg":
                    inputs.append(catalog[-1])
                    continue
                for data_obj in catalog:
                    if (data_obj not in inputs 
                    and (data_obj.format == format_spec or format_spec == DataFormat.ANY) 
                    and len(inputs) < len(task.input_spec)):
                        inputs.append(data_obj)
                        break
            
            # Generate outputs
            outputs = []
            for placeholder, format_spec in task.output_spec.items():
                if idx == len(self.tasks) - 1 and result:
                    outputs.append(result)
                    break
                else:
                    suffix = f"_{len(outputs)}"
                    output_path = gen_file_path(task, format_spec, prefix=f"{idx}_", suffix=suffix)
                    output_data = Data(path=output_path, format=format_spec)
                    outputs.append(output_data)
            
            catalog = outputs + catalog

            if len(inputs) != len(task.input_spec):
                missing_inputs = len(task.input_spec) - len(inputs)
                raise ValueError(
                    f"For task {task.name}: expected {task.input_spec} inputs, got {inputs}. "
                    f"Missing {missing_inputs} inputs."
                    f"catalog: {"\n".join([str(i) for i in catalog])}"
                )
            elif len(outputs) != len(task.output_spec):
                missing_outputs = len(task.output_spec) - len(outputs)
                raise ValueError(
                    f"\nFor task {task.name}: expected {task.output_spec} outputs, got {outputs}. "
                    f"\nMissing {missing_outputs} outputs."
                    f"\nCatalog: {"\n".join([str(i) for i in catalog])}"
                )
            else:
                print(f"Adding task '{task.name}' to plan with\n\t inputs: {[str(i.path) for i in inputs]} and \n\t outputs: {[str(o.path) for o in outputs]}")
                calls.append(KgPipePlanStep(
                    task=task.name, 
                    input=inputs, 
                    output=outputs
                ))
        
        self.plan = KgPipePlan(
            steps=calls,
            seed=self.seed,
            source=source,
            result=result
        )

        return self.plan

    def show(self) -> None:
        """Show the pipeline."""
        def custom_serializer(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        for stage in self.plan:
            print(json.dumps(stage, indent=4, default=custom_serializer))

    # TODO: Add a method to plot the pipeline
    def plot(self) -> None:
        """Plot the pipeline."""
        pass

    def run(self, stable_files_override: bool = False) -> List[KgTaskReport]:
        """Execute each task defined in the plan and collect the reports."""
        if not self.plan:
            raise ValueError("Pipeline plan is empty. Call build() first.")
        
        self.previous_was_skipped = True

        reports = []
        for task_spec in self.plan.steps:
            # Find the corresponding task
            task = next((t for t in self.tasks if t.name == task_spec.task), None)
            if not task:
                raise ValueError(f"Task {task_spec.task} not found in pipeline")
            
            logging.info(f"Running {task.name} with\n\t inputs: {[str(i.path) for i in task_spec.input]}\n\t outputs: {[str(o.path) for o in task_spec.output]}")

            # Validate inputs exist
            for input_data in task_spec.input:
                if not input_data.exists():
                    raise FileNotFoundError(f"Input file {input_data.path} does not exist")

            if self.previous_was_skipped:
                report = task.run(task_spec.input, task_spec.output, stable_files_override=stable_files_override)
            else:
                report = task.run(task_spec.input, task_spec.output, stable_files_override=True)

            if report.status != "skipped":
                self.previous_was_skipped = False

            reports.append(report)
        
        return reports
    
    def __str__(self) -> str:
        return f"KgPipe(tasks={len(self.tasks)}, target={self.seed})"

