"""
KGflex model.py
Core domain model for the KGflex framework.

This module defines the core data structures that represent the domain model
for generating and executing KG pipelines.
"""

from __future__ import annotations

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

# Format descriptions for built-in formats
FORMAT_DESCRIPTIONS = {
    "ttl": "Turtle RDF format",
    "nquads": "N-Quads RDF format",
    "json": "JSON format",
    "csv": "CSV format",
    "parquet": "Parquet format",
    "xml": "XML format",
    "rdf": "RDF format",
    "jsonld": "JSON-LD format",
    "txt": "Text format",
    "paris_csv": "Paris CSV format",
    "openrefine_json": "OpenRefine JSON format",
    "limes_xml": "LIMES XML format",
    "spotlight_json": "DBpedia Spotlight JSON format",
    "falcon_json": "FALCON JSON format",
    "ie_json": "Information Extraction JSON format",
    "valentine_json": "Valentine JSON format",
    "corenlp_json": "CoreNLP JSON format",
    "openie_json": "OpenIE JSON format",
    "agreementmaker_rdf": "AgreementMaker RDF format",
    "em_json": "Entity Matching JSON format",
}


class DataFormat(Enum):
    """Built-in data formats with enum benefits."""
    # Standard formats
    RDF_TTL = "ttl"
    RDF_NQUADS = "nq"
    RDF_NTRIPLES = "nt"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    RDF_XML = "xml"
    RDF = "rdf"
    RDF_JSONLD = "jsonld"
    TEXT = "txt"
    XML = "xml"
    ANY = "any"
    
    # Tool-specific formats
    PARIS_CSV = "paris.csv"
    OPENREFINE_JSON = "openrefine.json"
    LIMES_XML = "limes.xml"
    SPOTLIGHT_JSON = "spotlight.json"
    FALCON_JSON = "falcon.json"
    VALENTINE_JSON = "valentine.json"
    CORENLP_JSON = "corenlp.json"
    OPENIE_JSON = "openie.json"
    AGREEMENTMAKER_RDF = "agreementmaker.rdf"

    # Exchange formats
    ER_JSON = "er.json" # Entity Resolution JSON format
    TE_JSON = "te.json" # Text Extraction JSON format

    # LLM Tasks
    JSON_ONTO_MAPPING_JSON = "json_onto_mapping.json"

    @classmethod
    def from_extension(cls, extension: str) -> DataFormat:
        """Get a format by file extension."""
        return cls(extension.upper())
    
    @property
    def extension(self) -> str:
        """Get the file extension for this format."""
        return self.value
    
    @property
    def description(self) -> str:
        """Get the description for this format."""
        return FORMAT_DESCRIPTIONS.get(self.value, self.value)
    
    @property
    def is_tool_specific(self) -> bool:
        """Check if this is a tool-specific format."""
        tool_specific_formats = {
            "paris_csv", "openrefine_json", "limes_xml", "spotlight_json",
            "falcon_json", "ie_json", "valentine_json", "corenlp_json",
            "openie_json", "agreementmaker_rdf", "em_json"
        }
        return self.value in tool_specific_formats

    def __str__(self) -> str:
        return f".{self.value}"
    
    def __repr__(self) -> str:
        return f".{self.value}"


class DynamicFormat:
    """Dynamic format for submodules to register custom formats."""
    
    def __init__(self, name: str, extension: str, description: str, is_tool_specific: bool = False):
        self.name = name
        self.extension = extension
        self.description = description
        self.is_tool_specific = is_tool_specific

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler) -> Any:
        """Provide Pydantic schema for this type."""
        return core_schema.union_schema([
            core_schema.is_instance_schema(cls),
            core_schema.str_schema()
        ])
    
    @property
    def value(self) -> str:
        """Get the format value (same as name for compatibility)."""
        return self.name
    
    def __eq__(self, other) -> bool:
        """Compare formats by name."""
        if isinstance(other, DynamicFormat):
            return self.name == other.name
        elif isinstance(other, DataFormat):
            return self.name == other.value
        elif isinstance(other, str):
            return self.name == other
        return False
    
    def __hash__(self) -> int:
        """Hash based on name."""
        return hash(self.name)
    
    def __str__(self) -> str:
        return f"DynamicFormat({self.name})"
    
    def __repr__(self) -> str:
        return f"DynamicFormat(name='{self.name}', extension='{self.extension}', description='{self.description}', is_tool_specific={self.is_tool_specific})"


class FormatRegistry:
    """Registry for managing and discovering data formats."""
    
    _dynamic_formats: Dict[str, DynamicFormat] = {}
    
    @classmethod
    def register_format(cls, name: str, extension: str, description: str, is_tool_specific: bool = False) -> DynamicFormat:
        """Register a new dynamic data format."""
        if name in cls._dynamic_formats:
            return cls._dynamic_formats[name]
        
        format_obj = DynamicFormat(name, extension, description, is_tool_specific)
        cls._dynamic_formats[name] = format_obj
        return format_obj
    
    @classmethod
    def get_format(cls, name: str) -> Optional[Union[DataFormat, DynamicFormat]]:
        """Get a format by name, checking built-in formats first."""
        # Try built-in formats first
        try:
            return DataFormat(name)
        except ValueError:
            # Then check dynamic formats
            return cls._dynamic_formats.get(name)
    
    @classmethod
    def list_formats(cls, tool_specific_only: bool = False) -> List[Union[DataFormat, DynamicFormat]]:
        """List all registered formats."""
        formats = list(DataFormat) + list(cls._dynamic_formats.values())
        if tool_specific_only:
            formats = [f for f in formats if getattr(f, 'is_tool_specific', False)]
        return formats
    
    @classmethod
    def list_standard_formats(cls) -> List[Union[DataFormat, DynamicFormat]]:
        """List all standard (non-tool-specific) formats."""
        formats = list(DataFormat) + list(cls._dynamic_formats.values())
        return [f for f in formats if not getattr(f, 'is_tool_specific', False)]
    
    @classmethod
    def list_tool_specific_formats(cls) -> List[Union[DataFormat, DynamicFormat]]:
        """List all tool-specific formats."""
        formats = list(DataFormat) + list(cls._dynamic_formats.values())
        return [f for f in formats if getattr(f, 'is_tool_specific', False)]
    
    @classmethod
    def list_rdf_formats(cls) -> List[Union[DataFormat, DynamicFormat]]:
        """List all RDF formats."""
        rdf_formats = [DataFormat.RDF_TTL, DataFormat.RDF_NQUADS, DataFormat.RDF, DataFormat.RDF_JSONLD]
        dynamic_rdf = [f for f in cls._dynamic_formats.values() if 'rdf' in f.name.lower() or 'ttl' in f.name.lower()]
        return rdf_formats + dynamic_rdf
    
    @classmethod
    def list_text_formats(cls) -> List[Union[DataFormat, DynamicFormat]]:
        """List all text formats."""
        text_formats = [DataFormat.JSON, DataFormat.CSV, DataFormat.XML, DataFormat.TEXT]
        dynamic_text = [f for f in cls._dynamic_formats.values() if f.name.lower() in ['json', 'csv', 'xml', 'txt', 'yaml']]
        return text_formats + dynamic_text
    
    @classmethod
    def clear_dynamic_formats(cls) -> None:
        """Clear all dynamically registered formats (useful for testing)."""
        cls._dynamic_formats.clear()


# Type alias for any format
Format = Union[DataFormat, DynamicFormat]
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


class TaskStatus(Enum):
    """Status of a task in a pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Data(BaseModel):
    """Represents a data file with a specific format."""
    path: Path
    format: Format
    
    def __init__(self, *args, **data):
        # Handle positional arguments for backward compatibility
        if args:
            if len(args) == 2:
                data['path'] = args[0]
                data['format'] = args[1]
            elif len(args) == 1:
                data['path'] = args[0]
        
        if 'path' in data and isinstance(data['path'], str):
            data['path'] = Path(data['path'])
        super().__init__(**data)
    
    @field_validator('format', mode='before')
    @classmethod
    def validate_format(cls, v):
        """Convert string format to proper Format object."""
        if isinstance(v, str):
            # Try to convert string to DataFormat enum
            try:
                return DataFormat(v)
            except ValueError:
                # If it's not a DataFormat, it might be a DynamicFormat
                from .models import FormatRegistry
                dynamic_format = FormatRegistry.get_format(v)
                if dynamic_format:
                    return dynamic_format
                raise ValueError(f"Unknown format: {v}")
        return v
    
    def exists(self) -> bool:
        """Check if the data file exists."""
        return self.path.exists()

    def to_dict(self) -> Dict[str, str]:
        return {
            "path": str(self.path),
            "format": self.format.value
        }
    
    def __str__(self) -> str:
        return f"Data({self.path}, {self.format.value})"
    
    def __eq__(self, other):
        """Custom equality to handle format comparison."""
        if not isinstance(other, Data):
            return False
        return (self.path == other.path and 
                (hasattr(self.format, 'value') and hasattr(other.format, 'value') and 
                 self.format.value == other.format.value))




class KgTaskReport(BaseModel):
    """Report of a task execution."""
    task_name: str
    inputs: List[Data]
    outputs: List[Data]
    start_ts: float
    duration: float
    status: str
    error: Optional[str] = None
    
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

@dataclass
class KgTask:
    """Represents a task that can be executed in a pipeline."""
    name: str
    input_spec: Mapping[str, Format]
    output_spec: Mapping[str, Format]
    function: Callable[[Dict[str, Data], Dict[str, Data]], None]
    description: Optional[str] = None
    category: Optional[str] = None
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if not self.input_spec:
            raise ValueError("Input specification cannot be empty")
        if not self.output_spec:
            raise ValueError("Output specification cannot be empty")
        if not callable(self.function):
            raise ValueError("Function must be callable")

    def run(self, inputs: List[Data], outputs: List[Data], stable_files_override: bool = False) -> KgTaskReport:
        """Execute the task with given inputs and outputs."""
        start = time.time()
        try:
            named_inputs = self._match(inputs, self.input_spec)
            named_outputs = self._match(outputs, self.output_spec)

            # print(f"Running {self.name} with\n\t inputs: {[str(i.path) for i in named_inputs.values()]}\n\t outputs: {[str(o.path) for o in named_outputs.values()]}")
            print(f"Running {self.name} with\n\t inputs: {named_inputs}\n\t outputs: {named_outputs}")
            
            # Validate that all required inputs and outputs are present
            if len(named_inputs) != len(self.input_spec):
                missing = set(self.input_spec.keys()) - set(named_inputs.keys())
                available = {obj.format.value: obj for obj in inputs}
                expected = {k: v.value for k, v in self.input_spec.items()}
                raise ValueError(
                    f"Missing required inputs: {missing}. "
                    f"Expected: {expected}. "
                    f"Available: {[f'{obj.path} ({obj.format.value})' for obj in inputs]}"
                )
            
            if len(named_outputs) != len(self.output_spec):
                missing = set(self.output_spec.keys()) - set(named_outputs.keys())
                available = {obj.format.value: obj for obj in outputs}
                expected = {k: v.value for k, v in self.output_spec.items()}
                raise ValueError(
                    f"Missing required outputs: {missing}. "
                    f"Expected: {expected}. "
                    f"Available: {[f'{obj.path} ({obj.format.value})' for obj in outputs]}"
                )

            if stable_files_override:
                for output in named_outputs.values():
                    # delete the file or directory
                    if output.path.exists():
                        if output.path.is_file():
                            output.path.unlink()
                        elif output.path.is_dir():
                            shutil.rmtree(output.path)

            # if all outputs exists skip the task
            if all(output.path.exists() for output in named_outputs.values()):
                print(f"Skipping task {self.name} because all outputs exist")
                # exit(1)
                # TODO do not override old KgTaskReport
                return KgTaskReport(
                    task_name=self.name,
                    inputs=list(named_inputs.values()),
                    outputs=list(named_outputs.values()),
                    start_ts=start,
                    duration=time.time() - start,
                    status="skipped",
                )

            self.function(named_inputs, named_outputs)
            
            return KgTaskReport(
                task_name=self.name,
                inputs=list(named_inputs.values()),
                outputs=list(named_outputs.values()),
                start_ts=start,
                duration=time.time() - start,
                status="success",
            )

        except Exception as e:
            print(f"An error occurred while running the task '{self.name}'.")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            import traceback
            traceback.print_exc()
            return KgTaskReport(
                task_name=self.name,
                inputs=inputs,
                outputs=outputs,
                start_ts=start,
                duration=time.time() - start,
                status="failed",
                error=str(e)
            )

    @staticmethod
    def _match(data: List[Data], spec: Mapping[str, Format]) -> Dict[str, Data]:
        """Match data objects to specification by format."""
        res = {}
        for obj in data:
            for k, fmt in spec.items():
                if (obj.format == fmt or fmt == DataFormat.ANY) and k not in res:
                    res[k] = obj
                    break
        return res
    
    def __str__(self) -> str:
        return f"KgTask({self.name}, inputs={self.input_spec}, outputs={self.output_spec})"

    def __name__(self) -> str:
        return self.name


@dataclass
class DataSet:
    """Represents a dataset that can be used as input to a pipeline or stage."""
    id: str
    name: str
    path: Path
    format: Format
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
    
    def exists(self) -> bool:
        """Check if the dataset file exists."""
        return self.path.exists()
    
    def __str__(self) -> str:
        return f"DataSet({self.name}, {self.path}, {self.format.value})"

from rdflib import SKOS

@dataclass
class KG:
    """Represents a knowledge graph."""
    id: str
    name: str
    path: Path
    format: Format
    triple_count: Optional[int] = None
    entity_count: Optional[int] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph: Optional[Graph] = None
    data_graph: Optional[Graph] = None
    ontology_graph: Optional[Graph] = None
    plan: Optional[KgPipePlan] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if not self.name:
            raise ValueError("KG name cannot be empty")

    def get_graph(self) -> Graph:
        if self.graph is None:
            tmp = Graph().parse(self.path)
            graph = Graph()
            for s, p, o in tmp:
                if (str(p) != str(SKOS.altLabel)):
                    graph.add((s, p, o))
            self.graph = graph
        return self.graph

    def get_data_graph(self) -> Graph:
        return Graph()
    
    def get_ontology_graph(self) -> Graph:
        # TODO derive from graph
        if self.ontology_graph is None:
            self.ontology_graph = Graph()
        return self.ontology_graph

    def set_ontology_graph(self, graph: Graph) -> None:
        print(f"Setting ontology graph with {len(graph)} triples")
        self.ontology_graph = graph

    def exists(self) -> bool:
        """Check if the KG file exists."""
        return self.path.exists()
    
    def __str__(self) -> str:
        return f"KG({self.name}, {self.path}, {self.format.value})"


@dataclass
class Stage:
    """Represents a stage in a pipeline, containing one or more tasks."""
    id: str
    name: str
    description: Optional[str] = None
    tasks: List[KgTask] = field(default_factory=list)
    inputs: List[Union[DataSet, KG]] = field(default_factory=list)
    outputs: List[Union[DataSet, KG]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # List of stage IDs this stage depends on

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.name:
            raise ValueError("Stage name cannot be empty")
    
    def add_task(self, task: KgTask) -> None:
        """Add a task to this stage."""
        self.tasks.append(task)
    
    def __str__(self) -> str:
        return f"Stage({self.name}, tasks={len(self.tasks)})"


class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or name
    
    @abstractmethod
    def compute(self, ground_truth: KG, prediction: KG) -> float:
        """Compute the metric value."""
        pass
    
    def __str__(self) -> str:
        return f"Metric({self.name})"


@dataclass
class EvaluationReport:
    """Contains the results of evaluating a KG against a ground truth."""
    id: str
    ground_truth: KG
    prediction: KG
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric result to the report."""
        self.metrics[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            "id": self.id,
            "ground_truth": self.ground_truth.name,
            "prediction": self.prediction.name,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        return f"EvaluationReport({self.ground_truth.name} vs {self.prediction.name}, metrics={len(self.metrics)})"


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


# Backward compatibility aliases
Task = KgTask
Pipeline = KgPipe