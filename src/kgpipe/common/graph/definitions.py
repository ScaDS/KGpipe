from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any
from kgcore.api.kg import KGId

# Types #

type schema_format = str
type any_uri = str

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
    DataSpec = _NS["DataSpec"]
    DataEntity = _NS["Data"]
    DataType = _NS["DataType"]
    ConfigSpec = _NS["ConfigSpec"]
    ConfigBinding = _NS["ConfigBinding"]


    status = _NS["status"]
    started_at = _NS["started_at"]
    ended_at = _NS["ended_at"]
    schema = _NS["schema"]
    format = _NS["format"]
    name = _NS["name"]
    partOfTask = _NS["partOfTask"]
    hasSubtask = _NS["hasSubtask"]
    description = _NS["description"]

    version = _NS["version"]
    executesTask = _NS["executesTask"]
    supportsTask = _NS["supportsTask"]
    input = _NS["input"]
    output = _NS["output"]
    format = _NS["format"]
    config_spec = _NS["config_spec"]

    timestamp = _NS["timestamp"]
    version = _NS["version"]
    hash = _NS["hash"]
    size = _NS["size"]
    location = _NS["location"]
    data_type = _NS["data_type"]

    realisesTask = _NS["realisesTask"]
    usesImplementation = _NS["usesImplementation"]

    homepage = _NS["homepage"]
    implementsMethod = _NS["implementsMethod"]
    usesTool = _NS["usesTool"]
    hasParameter = _NS["hasParameter"]

    providesMethod = _NS["providesMethod"]

    key = _NS["key"]
    alias_keys = _NS["alias_keys"]
    datatype = _NS["datatype"]
    required = _NS["required"]
    default_value = _NS["default_value"]
    allowed_values = _NS["allowed_values"]
    minimum = _NS["minimum"]
    maximum = _NS["maximum"]
    unit = _NS["unit"]
    value = _NS["value"]
    binding = _NS["binding"]

    parameter = _NS["parameter"]
    hasParameterBinding = _NS["hasParameterBinding"]

# Entities #

DataTypeEntityId = KGId
class DataTypeEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    ### object properties ###
    format: str
    data_schema: str

DataEntityId = KGId
class DataEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    ### datatype properties ###
    timestamp: Optional[str] = None
    version: Optional[str] = None
    hash: Optional[str] = None
    size: Optional[int] = None
    ### object properties ###
    location: any_uri
    data_type: DataTypeEntityId

DataSpecEntityId = KGId
class DataSpecEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    name: str
    ### object properties ###
    data_type: DataTypeEntityId

TaskEntityId = KGId
class TaskEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    description: Optional[str] = None
    partOfTask: Optional[TaskEntityId] = None

# TODO MethodEntityId = KGId
# TODO class MethodEntity(BaseModel):
#     model_config = ConfigDict(frozen=True)
#     name: str
#     realizesTask: tuple[TaskEntityId, ...]

ToolEntityId = KGId
class ToolEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    ### datatype properties ###
    name: str
    homepage: Optional[str] = None
    ### object properties ###
    # NOTE: these entities are used as `lru_cache` keys; must be hashable.
    supportsTasks: tuple[TaskEntityId, ...]
    # TODO providesMethods: tuple[MethodEntityId, ...]

ParameterEntityId = KGId
class ParameterEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    key: str
    # NOTE: these entities are used as `lru_cache` keys; must be hashable.
    alias_keys: tuple[str, ...]
    datatype: str
    required: bool
    default_value: str | int | float | bool
    allowed_values: tuple[str | int | float | bool, ...]
    # description: Optional[str] = None
    # scope: Scope # (training/inference/io/resources)
    # constraints
    # minimum: Optional[float] = None
    # maximum: Optional[float] = None
    # unit: Optional[str] = None

ParameterBindingEntityId = KGId
class ParameterBindingEntity(BaseModel):
    value: Any
    parameter: ParameterEntityId

ConfigSpecEntityId = KGId
class ConfigSpecEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    name: str
    ### object properties ###
    # NOTE: these entities are used as `lru_cache` keys; must be hashable.
    parameters: tuple[ParameterEntityId, ...]

ConfigBindingEntityId = KGId
class ConfigBindingEntity(BaseModel):
    name: Any
    binding: tuple[ParameterBindingEntityId, ...]

ImplementationEntityId = KGId
class ImplementationEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    name: str
    version: str
    ### object properties ###
    input_spec: List[DataSpecEntityId]
    output_spec: List[DataSpecEntityId]
    realizesTask: List[TaskEntityId]
    usesTool: List[ToolEntityId]
    config_spec: Optional[ConfigSpecEntityId] = None

    # TODO implementsMethod: List[MethodEntityId]
    # TODO interface: str

TaskRunEntityId = KGId
class TaskRunEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    status: str
    started_at: float
    ended_at: float
    ### object properties ###
    input: List[DataEntityId]
    output: List[DataEntityId]
    # TODO executesTask: TaskEntityId
    usesImplementation: ImplementationEntityId
    hasConfigBinding: Optional[ConfigBindingEntityId] = None

# Entity representing a task dag (not the implementation)
# class PipelineDefinitionEntity(BaseModel):
#     """
#     The definition of a pipeline
#     """
#     placeholder: str
#     #definesPipeline: Pipeline

PipelineStepEntityId = KGId
class PipelineStepEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    name: str
    ### object properties ###
    input: List[DataEntityId]
    output: List[DataEntityId]
    executesTask: TaskEntityId

# TODO issue as the Graph has no ordering of the tasks
class PipelineEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    name: str
    ### object properties ###
    steps: List[PipelineStepEntityId]
    firstStep: PipelineStepEntityId
    lastStep: PipelineStepEntityId
    input: List[DataEntityId]
    output: List[DataEntityId]

PipelineRunEntityId = KGId
class PipelineRunEntity(BaseModel):
    """
    The result of a pipeline execution
    """
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    name: str
    status: str
    started_at: float
    ended_at: float
    ### object properties ###
    hasTaskRun: List[TaskRunEntity]
    # TODO usesPipelineDefinition: PipelineDefinition
    # TODO runsPipeline: PipelineStepEntityId

MetricEntityId = KGId
class MetricEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    ### datatype properties ###
    name: str
    description: Optional[str] = None
    type: str # TODO should be an enum
    ### object properties ###
    # TODO output: List[schema_format]
    # TODO hasParameter: List[ParameterId]

MetricRunEntityId = KGId
class MetricRunEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: Optional[str] = None
    ### datatype properties ###
    status: str
    started_at: float
    ended_at: float
    value: float
    details: str # TODO should be a dictionary
    ### object properties ###
    computedMetric: MetricEntityId
    input: List[DataEntityId]
