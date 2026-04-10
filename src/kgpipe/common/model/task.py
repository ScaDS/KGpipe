from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Mapping, Optional, Callable
# import field
from dataclasses import dataclass, field
from .data import Data, Format, DataFormat
from pydantic import BaseModel, Field, ConfigDict, model_validator
import time
import shutil
from uuid import uuid4
import inspect
from kgpipe.common.model.default_catalog import TaskCategory
from .configuration import (
    Parameter,
    ConfigurationDefinition,
    ConfigurationProfile,
    ParameterType,
)
from kgpipe.common.graph.systemgraph import PipeKG
from kgpipe.common.graph.mapper import task_run_to_entity

type TaskName = str

type TaskInput = Dict[TaskName, Data]
type TaskOutput = Dict[TaskName, Data]

class KgTaskReport(BaseModel):
    """Report of a task execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Backwards-compatible identifier for persisted reports (`exec-report.json`).
    # Historically we stored only the task name; newer runtime code may also attach the `KgTask`.
    task_name: str
    task: Optional["KgTask"] = Field(default=None, exclude=True)
    inputs: List[Data]
    outputs: List[Data]
    start_ts: float
    duration: float
    status: str
    error: Optional[str] = None
    config_profile: Optional[ConfigurationProfile] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_task_fields(cls, data):
        """
        Accept both legacy reports (with `task_name`) and new runtime reports (with `task`).
        """
        if not isinstance(data, dict):
            return data

        # If we have a task object but no explicit task_name, derive it.
        if "task_name" not in data and "task" in data and data["task"] is not None:
            task_obj = data["task"]
            name = getattr(task_obj, "name", None)
            if name is not None:
                data["task_name"] = name

        return data

KgTaskRun = KgTaskReport

class TaskStatus(Enum):
    """Status of a task in a pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

# # TODO impl later for typed api
# class TaskCatalog():
#     pass

@dataclass
class KgTask:
    """Represents a task that can be executed in a pipeline."""
    name: str
    input_spec: Mapping[str, Format]
    output_spec: Mapping[str, Format]
    function: Callable[[Dict[str, Data], Dict[str, Data]], None]
    description: Optional[str] = None
    category: List[TaskCategory] = field(default_factory=list)
    config_spec: Optional[ConfigurationDefinition] = None
    tools: List[str] = field(default_factory=list)
    trace_task_run: bool = False
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if not self.input_spec:
            raise ValueError("Input specification cannot be empty")
        if not self.output_spec:
            raise ValueError("Output specification cannot be empty")
        if not callable(self.function):
            raise ValueError("Function must be callable")


    # TODO if configProfile is not provided, use the default config profile derived from the config_spec
    def run(self, inputs: List[Data], outputs: List[Data], stable_files_override: bool = False, configProfile: Optional[ConfigurationProfile] = None) -> KgTaskReport:
        """Execute the task with given inputs and outputs."""
        start = time.time()
        report: KgTaskReport
        try:
            named_inputs = self._match(inputs, self.input_spec)
            named_outputs = self._match(outputs, self.output_spec)
            print(f"Running {self.name} with\n\t inputs: {named_inputs}\n\t outputs: {named_outputs}")

            self._validate_required_data(named_inputs, self.input_spec, "inputs", inputs)
            self._validate_required_data(named_outputs, self.output_spec, "outputs", outputs)
            self._prepare_outputs(named_outputs, stable_files_override)

            # TODO needs to check config profile changes, or maybe not
            if self._should_skip(named_outputs):
                print(f"Skipping task {self.name} because all outputs exist")
                report = self._build_report(start, "skipped", list(named_inputs.values()), list(named_outputs.values()), config_profile=configProfile)
                self._trace_task_run_to_pipekg(report)
                return report

            self._call_function(named_inputs, named_outputs, configProfile)
            report = self._build_report(start, "success", list(named_inputs.values()), list(named_outputs.values()), config_profile=configProfile)
            self._trace_task_run_to_pipekg(report)
            return report

        except Exception as e:
            print(f"An error occurred while running the task '{self.name}'.")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            import traceback
            traceback.print_exc()
            report = self._build_report(start, "failed", inputs, outputs, error=str(e), config_profile=configProfile)
            self._trace_task_run_to_pipekg(report)
            return report

    def _trace_task_run_to_pipekg(self, report: KgTaskReport) -> None:
        # TODO print(f"Tracing task run to pipekg: {report}")
        if not self.trace_task_run:
            return
        task_run_to_entity(report)

    def _call_function(
        self,
        named_inputs: Dict[str, Data],
        named_outputs: Dict[str, Data],
        config_profile: Optional[object],
    ) -> None:
        """
        Call the wrapped task function with or without config.

        Supported task signatures:
        - fn(inputs, outputs)
        - fn(inputs, outputs, config)
        - fn(inputs, outputs, *, config=...)
        - fn(inputs, outputs, **kwargs)  (will receive config=... if provided)
        """
        sig = inspect.signature(self.function)
        params = sig.parameters

        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        has_config_param = "config" in params

        if config_profile is None:
            # If config is required positionally/without default, fail early with a clear error.
            if has_config_param:
                p = params["config"]
                if p.default is inspect._empty and p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    raise TypeError(
                        f"{self.name} requires a 'config' argument but none was provided. "
                        f"Pass configProfile=... to KgTask.run(), or make 'config' optional."
                    )
            self.function(named_inputs, named_outputs)
            return

        # config is provided: pass it only if the function can accept it
        if has_config_param or accepts_var_kwargs:
            # If the task declares a config spec, we require a structured ConfigurationProfile.
            if self.config_spec is not None and not isinstance(config_profile, ConfigurationProfile):
                raise TypeError(
                    f"{self.name} expects configProfile to be a ConfigurationProfile "
                    f"because it declares config_spec='{self.config_spec.name}', "
                    f"got {type(config_profile).__name__}."
                )
            if isinstance(config_profile, ConfigurationProfile) and self.config_spec is not None:
                self._validate_config(config_profile, self.config_spec)
            self.function(named_inputs, named_outputs, config=config_profile)
            return

        # Function cannot accept config: ignore it
        self.function(named_inputs, named_outputs)

    def _validate_config(self, config_profile: ConfigurationProfile, config_spec: ConfigurationDefinition) -> None:
        if config_profile.definition.name != config_spec.name:
            raise ValueError(
                f"Config profile definition '{config_profile.definition.name}' does not match "
                f"task config spec '{config_spec.name}'."
            )

        spec_by_name: Dict[str, Parameter] = {p.name: p for p in config_spec.parameters}
        spec_by_key: Dict[str, Parameter] = {}
        for p in config_spec.parameters:
            spec_by_key[p.name] = p
            for nk in p.native_keys:
                spec_by_key[nk] = p

        bound: Dict[str, object] = {}
        for binding in config_profile.bindings:
            raw_key = binding.parameter.name
            if raw_key not in spec_by_key:
                raise ValueError(
                    f"Unknown config parameter '{raw_key}' for spec '{config_spec.name}'. "
                    f"Known: {sorted(spec_by_name.keys())}"
                )
            param = spec_by_key[raw_key]
            value = binding.value
            bound[param.name] = value

            if param.datatype == ParameterType.boolean and not isinstance(value, bool):
                raise TypeError(f"Config parameter '{param.name}' expects boolean, got {type(value).__name__}")
            if param.datatype == ParameterType.integer and not isinstance(value, int):
                raise TypeError(f"Config parameter '{param.name}' expects integer, got {type(value).__name__}")
            if param.datatype == ParameterType.number and not isinstance(value, (int, float)):
                raise TypeError(f"Config parameter '{param.name}' expects number, got {type(value).__name__}")
            if param.datatype == ParameterType.string and not isinstance(value, str):
                raise TypeError(f"Config parameter '{param.name}' expects string, got {type(value).__name__}")

            if param.allowed_values and value not in param.allowed_values:
                raise ValueError(
                    f"Config parameter '{param.name}' value {value!r} not in allowed_values {param.allowed_values!r}"
                )

            if param.minimum is not None:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Config parameter '{param.name}' has minimum constraint but value is not numeric")
                if value < param.minimum:
                    raise ValueError(f"Config parameter '{param.name}' value {value} < minimum {param.minimum}")

            if param.maximum is not None:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Config parameter '{param.name}' has maximum constraint but value is not numeric")
                if value > param.maximum:
                    raise ValueError(f"Config parameter '{param.name}' value {value} > maximum {param.maximum}")

        missing_required: List[str] = []
        for p in config_spec.parameters:
            if not p.required:
                continue
            if p.name in bound:
                continue
            if getattr(p, "default_value", None) is None:
                missing_required.append(p.name)
        if missing_required:
            raise ValueError(f"Missing required config parameters: {missing_required}")


    def _build_report(
        self,
        start_ts: float,
        status: str,
        inputs: List[Data],
        outputs: List[Data],
        error: Optional[str] = None,
        config_profile: Optional[ConfigurationProfile] = None,
    ) -> KgTaskReport:
        return KgTaskReport(
            task=self,
            task_name=self.name,
            inputs=inputs,
            outputs=outputs,
            start_ts=start_ts,
            duration=time.time() - start_ts,
            status=status,
            error=error,
            config_profile=config_profile,
        )

    def _validate_required_data(
        self,
        matched: Dict[str, Data],
        spec: Mapping[str, Format],
        label: str,
        raw_items: List[Data],
    ) -> None:
        if len(matched) == len(spec):
            return

        missing = set(spec.keys()) - set(matched.keys())
        expected = {k: v.value for k, v in spec.items()}
        available = [f"{obj.path} ({obj.format.value})" for obj in raw_items]
        raise ValueError(
            f"Missing required {label}: {missing}. "
            f"Expected: {expected}. "
            f"Available: {available}"
        )

    def _prepare_outputs(self, outputs: Dict[str, Data], stable_files_override: bool) -> None:
        if not stable_files_override:
            return
        for output in outputs.values():
            if output.path.exists():
                if output.path.is_file():
                    output.path.unlink()
                elif output.path.is_dir():
                    shutil.rmtree(output.path)

    def _should_skip(self, outputs: Dict[str, Data]) -> bool:
        return all(output.path.exists() for output in outputs.values())

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