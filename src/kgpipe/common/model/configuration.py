from typing import List, Any, Optional
from enum import Enum

class Scope(Enum):
    training = "training"
    inference = "inference"
    io = "io"
    resources = "resources"
    general = "general"

class ParameterDataType(str, Enum):
    boolean = "boolean"
    integer = "integer"
    number = "number"      # float
    string = "string"
    enum = "enum"
    array = "array"
    object = "object"

# class ParameterSetting():
#     pass

# class OptionCategory():
#     pass

# class CanonicalOption():
#     pass

# class ParameterBinding():
#     pass

class Parameter():
    #   +id: string
    #   +name: string
    #   +native_keys: string[*]
    #   +datatype: DataType
    #   +default_value: any?
    #   +required: bool
    #   +scope: Scope
    #   +allowed_values: any[*]?
    #   +min/max/unit: number?/number?/string?
    id: str
    name: str
    native_keys: List[str]
    datatype: ParameterDataType
    default_value: Any
    required: bool
    scope: Scope
    allowed_values: List[Any]
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None

# class ConfigurationProfile():
#     pass