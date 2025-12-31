from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
# from kgcore.api.kg import KnowledgeGraph, KGProperty
# from kgcore.backend.rdf import RDFLibBackend
# from kgcore.model.rdf import RDFBaseModel
# from kgcore.system import SystemRecorder, set_default_recorder, class_, event, pydantic_model

# TODO add annotations to the classes here

# Types #

type schema_format = str

# Data #

class DataHandle(BaseModel):
    """
    A handle to a data artifact

    uri: file://example.com/data.txt
    type: any/text
    timestamp: 2021-01-01
    version: 1.0.0
    hash: 1234567890
    size: 1000
    """
    uri: str
    type: schema_format
    timestamp: Optional[str] = None
    version: Optional[str] = None
    hash: Optional[str] = None
    size: Optional[int] = None

# Task #

# TODO describing entity vs entity with used values for the task
class TaskConfiguration(BaseModel):
    key: str
    value: Any
    
class Task(BaseModel):
    """
    A function that implements a task in a pipeline

    name: paris_rdf_matcher
    type: entity_resolution
    description: "PARIS java implementation to match two RDF files, producing CSV files..."
    input: [any_rdf, any_rdf]
    output: [any_csv]
    """
    name: str
    type: str
    description: Optional[str] = None
    input: List[schema_format]
    output: List[schema_format]

class TaskResult(BaseModel):
    """
    The result of a task execution including configuration variables
    """
    task: Task
    config: Dict[str, Any]
    input: List[DataHandle]
    output: List[DataHandle]
    status: str
    duration: float

# Evaluation #

class Eval(BaseModel):
    """
    A function that evaluates data produced by tasks
    """
    name: str
    type: str
    description: Optional[str] = None
    input: List[schema_format]

class EvalResult(BaseModel):#
    """
    Result of an evaluation function
    """
    eval: Eval
    config: Dict[str, Any]
    input: List[DataHandle]
    output: Dict[str, Any]
    status: str
    duration: float

# Pipeline #

class Pipeline(BaseModel):
    """
    The plan of a pipeline
    """
    tasks: List[Task]
    input: List[schema_format]
    output: List[schema_format]

class PipelineResult(BaseModel):
    """
    Result of a pipeline execution
    """
    task_results: List[TaskResult]
    eval_results: List[EvalResult]
    input: List[DataHandle]
    output: List[DataHandle]
    status: str
    duration: float