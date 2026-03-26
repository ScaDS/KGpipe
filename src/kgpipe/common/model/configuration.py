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


class ParameterBinding(BaseModel):
    """
    Binding of a configuration parameter to a value in the pipeline execution
    """
    parameter: Parameter
    value: str | int | float | bool # TODO extend to more types?


class ConfigurationDefinition(BaseModel):
    """
    Possible configurations specification of a task
    """
    name: str
    description: Optional[str] = None
    parameters: List[Parameter] = field(default_factory=list)


class ConfigurationProfile(BaseModel):
    """
    Configuration profile specification, the actual values of the parameters in the pipeline execution
    """
    name: str
    definition: ConfigurationDefinition
    description: Optional[str] = None   
    bindings: List[ParameterBinding] = field(default_factory=list)

    def get_parameter(self, name: str) -> Parameter:
        for parameter in self.definition.parameters:
            if parameter.name == name:
                return parameter
        raise ValueError(f"Parameter {name} not found in configuration profile {self.name}")

    def get_parameter_binding(self, name: str) -> ParameterBinding:
        for binding in self.bindings:
            if binding.parameter.name == name:
                return binding
        raise ValueError(f"Parameter binding {name} not found in configuration profile {self.name}")

    def get_parameter_value(self, name: str) -> str | int | float | bool:
        return self.get_parameter_binding(name).value

class ConfigurationBuilder():
    def __init__(self, config_spec: ConfigurationDefinition):
        self.config_spec = config_spec
        self.config_profile = ConfigurationProfile(name=config_spec.name, definition=config_spec)

    def add_parameter(self, name: str, value: str | int | float | bool) -> None:
        self.config_profile.bindings.append(ParameterBinding(parameter=self.get_parameter(name), value=value))



class ConfigurationMapping(BaseModel):
    """
    Mapping of a configuration profile to a task implementation
    """
    for_task_spec: ConfigurationDefinition
    to_global_spec: ConfigurationDefinition