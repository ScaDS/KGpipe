from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Mapping, Optional, Callable
# import field
from dataclasses import dataclass, field
from .data import Data, Format, DataFormat
from pydantic import BaseModel
import time
import shutil

from .configuration import Parameter, ConfigurationProfile
from kgpipe.common.annotations import kg_class

type TaskName = str

type TaskInput = Dict[TaskName, Data]
type TaskOutput = Dict[TaskName, Data]

@kg_class()
class KgTaskReport(BaseModel):
    """Report of a task execution."""
    task_name: str
    inputs: List[Data]
    outputs: List[Data]
    start_ts: float
    duration: float
    status: str
    error: Optional[str] = None

class TaskStatus(Enum):
    """Status of a task in a pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

# TODO impl later for typed api
class TaskCategory():
    pass

# TODO impl later for typed api
class TaskCatalog():
    pass

@kg_class()
@dataclass
class KgTask:
    """Represents a task that can be executed in a pipeline."""
    name: str
    input_spec: Mapping[str, Format]
    output_spec: Mapping[str, Format]
    function: Callable[[Dict[str, Data], Dict[str, Data]], None]
    description: Optional[str] = None
    category: List[str] = field(default_factory=list)
    config: Optional[ConfigurationProfile] = None
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if not self.input_spec:
            raise ValueError("Input specification cannot be empty")
        if not self.output_spec:
            raise ValueError("Output specification cannot be empty")
        if not callable(self.function):
            raise ValueError("Function must be callable")

    def run(self, inputs: List[Data], outputs: List[Data], stable_files_override: bool = False, configProfile: Optional[str] = None) -> KgTaskReport:
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