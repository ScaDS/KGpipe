from typing import List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel

from kgpipe.common.annotations import kg_class

# TODO register in syskg as Datatype
class ParameterType(Enum):
    """
    Built-in parameter types of configuration parameters
    """
    boolean = "boolean"
    integer = "integer"
    number = "number"      # float
    string = "string"
    enum = "enum"
    array = "array"
    object = "object"


@kg_class()
class Parameter(BaseModel):
    """
    Configuration parameter definition, not the actual value of the parameter in the pipeline execution
    """
    #   +id: string
    #   +name: string
    #   +native_keys: string[*]
    #   +datatype: DataType
    #   +default_value: any?
    #   +required: bool
    #   +scope: Scope
    #   +allowed_values: any[*]?
    #   +min/max/unit: number?/number?/string?
    name: str
    native_keys: List[str]
    datatype: ParameterType
    default_value: str | int | float | bool
    required: bool
    # scope: Scope # (training/inference/io/resources)
    # constraints
    allowed_values: List[str | int | float | bool]
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None


@kg_class()
class ParameterBinding(BaseModel):
    """
    Binding of a configuration parameter to a value in the pipeline execution
    """
    parameter: Parameter
    value: str | int | float | bool # TODO extend to more types?

    
@kg_class()
class ConfigurationProfile(BaseModel):
    """
    Configuration profile definition, not the actual values of the parameters in the pipeline execution
    """
    name: str
    description: Optional[str] = None   
    bindings: List[ParameterBinding] = field(default_factory=list)