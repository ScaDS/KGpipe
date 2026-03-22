from dataclasses import dataclass
from sys import implementation
from pydantic import BaseModel
from typing import Mapping, Optional, List, Dict, Any
from kgcore.api.kg import KGId

from kgpipe.common.model.data import DataFormat

# Types #

type schema_format = str

# Vocabulary #

from rdflib.namespace import DefinedNamespace, Namespace

class KGPIPE_NS(DefinedNamespace):
    _fail = True
    _NS = Namespace("http://github.com/ScaDS/kgpipe/")
    Task = _NS["Task"]
    TaskRun = _NS["TaskRun"]
    Method = _NS["Method"]
    Tool = _NS["Tool"]
    Implementation = _NS["Implementation"]
    Parameter = _NS["Parameter"]
    ParameterBinding = _NS["ParameterBinding"]
    Pipeline = _NS["Pipeline"]
    PipelineRun = _NS["PipelineRun"]
    Artifact = _NS["Artifact"]
    ArtifactType = _NS["ArtifactType"]
    Schema = _NS["Schema"]
    Metric = _NS["Metric"]
    MetricRun = _NS["MetricRun"]

# Entities #

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

TaskEntityId = KGId
class TaskEntity(BaseModel):
    name: str
    hasSubtask: List[TaskEntityId]

MethodEntityId = KGId
class MethodEntity(BaseModel):
    name: str
    realizesTask: List[TaskEntityId]
    
ToolEntityId = KGId
class ToolEntity(BaseModel):
    name: str
    # supportsTasks: List[Task]
    providesMethods: List[MethodEntityId]

ParameterId = KGId
class ParameterEntity(BaseModel):
    name: str
    value: Any
    type: str
    description: Optional[str] = None
    default_value: Optional[Any] = None
    required: bool = False
    allowed_values: Optional[List[Any]] = None

ParameterBindingId = KGId
class ParameterBindingEntity(BaseModel):
    value: Any
    parameter: ParameterId

ImplementationEntityId = KGId
class ImplementationEntity(BaseModel):
    uri: Optional[str] = None
    name: str
    input_spec: List[str]
    output_spec: List[str]
    implementsMethod: List[MethodEntityId]
    hasParameter: List[ParameterId]
    usesTool: List[ToolEntityId]

    # interface: str # TODO: Interface    
    # hasParameter: Parameter

TaskRunEntityId = KGId
class TaskRunEntity(BaseModel):
    number: int
    name: str
    status: str
    started_at: float
    ended_at: float
    input: List[DataHandle]
    output: List[DataHandle]
    executesTask: TaskEntityId
    usesImplementation: ImplementationEntityId
    hasParameterBinding: List[ParameterBindingId]

# Entity representing a task dag (not the implementation)
# class PipelineDefinitionEntity(BaseModel):
#     """
#     The definition of a pipeline
#     """
#     placeholder: str
#     #definesPipeline: Pipeline

# TODO issue as the Graph has no ordering of the tasks
class PipelineEntity(BaseModel):
    name: str
    tasks: List[TaskEntityId]
    input: List[DataHandle]
    output: List[DataHandle]

class PipelineRunEntity(BaseModel):
    """
    The result of a pipeline execution
    """
    name: str
    status: str
    started_at: float
    ended_at: float
    hasTaskRun: List[TaskRunEntity]
    # usesPipelineDefinition: PipelineDefinition
    # runsPipeline: Pipeline

MetricEntityId = KGId
class MetricEntity(BaseModel):
    name: str
    description: Optional[str] = None
    type: str
    # output: List[schema_format]
    # hasParameter: List[ParameterId]

MetricRunEntityId = KGId
class MetricRunEntity(BaseModel):
    status: str
    started_at: float
    ended_at: float
    computedMetric: MetricEntityId
    input: List[DataHandle]
    value: float
    details: str