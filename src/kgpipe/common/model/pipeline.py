import os
import time
import uuid
import hashlib
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
from kgcore.api.kg import KGId
from rdflib import Graph
from pydantic import BaseModel, Field, field_validator
from pydantic_core import core_schema

from .data import Data, DataFormat, DataSet, Format
from .task import KgTask, KgTaskReport
from .configuration import ConfigurationProfile
from kgpipe.generation.spec import (
    PipelinePortDecl,
    PipelineTaskConf,
    candidate_fingerprint,
    profile_parameter_values,
)
# from .kg import KG
from kgpipe.common.annotations import kg_class
from kgpipe.common.graph.systemgraph import PipeKG


def _format_name(fmt: Format) -> str:
    return str(getattr(fmt, "value", fmt))


def _format_matches(data_obj: Data, format_spec: Format) -> bool:
    return data_obj.format == format_spec or format_spec == DataFormat.ANY


def _ref_for(data_obj: Data, ref_by_id: Dict[int, str]) -> str:
    existing = ref_by_id.get(id(data_obj))
    if existing is not None:
        return existing
    generated = f"$data:{data_obj.path}"
    ref_by_id[id(data_obj)] = generated
    return generated


class KgPipePlanStep(BaseModel):
    """A step in a KG pipeline plan.

    ``input`` and ``output`` are the concrete files. ``inputs`` and ``outputs``
    name each port and the ref it was bound to (``$source``, ``$seed``,
    ``$result``, or ``step_id.port``).
    """
    task: str
    input: List[Data]
    output: List[Data]
    id: str = ""
    inputs: Dict[str, str] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)

# kg_class()
class KgPipePlan(BaseModel):
    """A resolved KG pipeline plan.

    ``fingerprint`` covers task identity, port refs, and parameter values.
    It does not cover file paths.
    """
    steps: List[KgPipePlanStep]
    seed: Optional[Data] = None
    source: Optional[Data] = None
    result: Optional[Data] = None
    fingerprint: str = ""
    connect: str = "auto"

    @staticmethod
    def from_path(json_file: str) -> 'KgPipePlan':
        with open(json_file, "r") as f:
            json_data = json.load(f)
            return KgPipePlan(**json_data)

    # def __str__(self) -> str:
    #     return f"KgTaskReport({self.task_name}, {self.status}, {self.duration:.2f}s)"

# TODO rename to KgPipeReport
# @kg_class()
class KgStageReport(BaseModel):
    """Report of a stage execution."""
    stage_name: str
    task_reports: List[KgTaskReport]
    start_ts: float
    duration: float
    status: str
    error: Optional[str] = None

    @staticmethod
    def from_path(json_file: str) -> 'KgStageReport':
        with open(json_file, "r") as f:
            json_data = json.load(f)
            return KgStageReport(**json_data)

KgPipeReport = KgStageReport
KgPipelineRun = KgStageReport

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


# TODO rename to Pipeline
# @kg_class()
@dataclass
class KgPipe:
    """Execution object for a list of tasks.

    New pipelines should be defined with :class:`kgpipe.api.Pipeline`. That
    call compiles a ``kgpipe.pipeline/v1`` specification into a ``KgPipe``.
    Constructing ``KgPipe`` directly still builds and runs.
    """
    
    tasks: List[KgTask]
    seed: Data
    data_dir: str = ""
    name: str = "Unknown"
    data: List[Data] = field(default_factory=list)
    plan: KgPipePlan = field(default_factory=lambda: KgPipePlan(
        steps=[],
        seed=None,
        source=None,
        result=None
    ))
    connect: str = "auto"
    step_specs: List[PipelineTaskConf] = field(default_factory=list)
    config_catalog: Dict[str, ConfigurationProfile] = field(default_factory=dict)
    input_decls: Dict[str, PipelinePortDecl] = field(default_factory=dict)
    output_decls: Dict[str, PipelinePortDecl] = field(default_factory=dict)

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


    def build(
        self,
        source: Data,
        result: Optional[Data] = None,
        stable_files: bool = False,
        configCatalog: Optional[Mapping[str, ConfigurationProfile]] = None,
    ) -> KgPipePlan:
        """Resolve task ports into an execution plan.

        ``connect: auto`` (the default) binds ports from a catalog that starts
        as ``[source, seed]``. A port named ``kg`` takes the last catalog entry,
        which stays the seed because new outputs are prepended. Every other
        port takes the first unused entry whose format matches. ``connect:
        explicit`` uses the refs on ``step_specs`` instead.
        """
        self._check_declared_formats(source, result)
        catalog = [source] + self.data
        calls: List[KgPipePlanStep] = []
        steps = self._steps_for_build()
        active_profiles = self.config_catalog if configCatalog is None else configCatalog
        ref_by_id: Dict[int, str] = {id(source): "$source", id(self.seed): "$seed"}
        if result is not None and result is not source and result is not self.seed:
            ref_by_id[id(result)] = "$result"
        symbols: Dict[str, Data] = {"$source": source, "$seed": self.seed}
        if result is not None:
            symbols["$result"] = result
        result_used = False

        def _profile_fingerprint(profile: Optional[ConfigurationProfile]) -> str:
            if profile is None:
                return ""
            # Make it stable regardless of binding order.
            bindings = []
            for b in getattr(profile, "bindings", []) or []:
                param = getattr(b, "parameter", None)
                pname = getattr(param, "name", None)
                if pname is None:
                    pname = str(param)
                bindings.append((str(pname), b.value))
            bindings.sort(key=lambda kv: kv[0])
            payload = json.dumps(
                {"definition": getattr(getattr(profile, "definition", None), "name", None), "bindings": bindings},
                sort_keys=True,
                default=str,
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        def _chain_hash(prev_hash: str, task_name: str, profile: Optional[ConfigurationProfile]) -> str:
            fp = _profile_fingerprint(profile)
            payload = json.dumps({"prev": prev_hash, "task": task_name, "profile": fp}, sort_keys=True)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        prev_hash = "0" * 64

        def gen_file_path(
            *,
            task: KgTask,
            format_spec: Format,
            prefix: str = "",
            suffix: str = "",
            task_hash: Optional[str] = None,
        ) -> Path:
            # Backwards-compatible: stable_files without configCatalog keeps the old deterministic names.
            if stable_files and configCatalog is None:
                return Path(self.data_dir) / f"{prefix}{task.name}{suffix}.{format_spec.extension}"

            # If configCatalog is provided, filenames must be deterministic based on the hash chain.
            if configCatalog is not None and task_hash is not None:
                short = task_hash[:12]
                return Path(self.data_dir) / f"{prefix}{task.name}.{short}{suffix}.{format_spec.extension}"

            # Default behavior: unique filenames.
            return Path(self.data_dir) / f"{prefix}{task.name}.{uuid4().hex}.{format_spec.extension}"
                

        for idx, task in enumerate(self.tasks):
            step = steps[idx]
            step_id = step.id or task.name
            task_hash: Optional[str] = None
            if configCatalog is not None:
                profile = configCatalog.get(step_id) or configCatalog.get(task.name)
                task_hash = _chain_hash(prev_hash, task.name, profile)
                prev_hash = task_hash

            resolved_profile = None
            if active_profiles:
                resolved_profile = active_profiles.get(step_id) or active_profiles.get(task.name)
            parameters = profile_parameter_values(resolved_profile)

            if self.connect == "explicit":
                inputs, input_refs, outputs, output_refs, result_used = self._bind_explicit_step(
                    task=task,
                    step=step,
                    step_id=step_id,
                    idx=idx,
                    result=result,
                    symbols=symbols,
                    ref_by_id=ref_by_id,
                    result_used=result_used,
                    gen_file_path=gen_file_path,
                    task_hash=task_hash,
                )
            else:
                inputs, input_refs = self._bind_auto_inputs(task, catalog, ref_by_id)
                outputs, output_refs = self._bind_auto_outputs(
                    task=task,
                    step_id=step_id,
                    idx=idx,
                    result=result,
                    ref_by_id=ref_by_id,
                    gen_file_path=gen_file_path,
                    task_hash=task_hash,
                )
                catalog = outputs + catalog

            if len(inputs) != len(task.input_spec):
                bound = set(input_refs)
                missing = [port for port in task.input_spec if port not in bound]
                available = [
                    f"{_ref_for(item, ref_by_id)} ({_format_name(item.format)})"
                    for item in catalog
                ]
                expected = {
                    port: _format_name(fmt) for port, fmt in task.input_spec.items()
                }
                raise ValueError(
                    f"Cannot connect task '{task.name}': missing input port(s) {missing}. "
                    f"Expected {expected}. Available: {available or '(none)'}."
                )
            elif len(outputs) != len(task.output_spec):
                expected = {
                    port: _format_name(fmt) for port, fmt in task.output_spec.items()
                }
                raise ValueError(
                    f"Cannot connect task '{task.name}': expected outputs {expected}, "
                    f"got {len(outputs)}."
                )
            else:
                print(f"Adding task '{task.name}' to plan with\n\t inputs: {[str(i.path) for i in inputs]} and \n\t outputs: {[str(o.path) for o in outputs]}")
                calls.append(KgPipePlanStep(
                    task=task.name,
                    id=step_id,
                    input=inputs,
                    output=outputs,
                    inputs=input_refs,
                    outputs=output_refs,
                    parameters=parameters,
                ))

        fingerprint = candidate_fingerprint([
            {
                "id": step.id,
                "task": step.task,
                "inputs": step.inputs,
                "outputs": step.outputs,
                "parameters": step.parameters,
            }
            for step in calls
        ])
        self.plan = KgPipePlan(
            steps=calls,
            seed=self.seed,
            source=source,
            result=result,
            fingerprint=fingerprint,
            connect=self.connect,
        )

        return self.plan

    def _steps_for_build(self) -> List[PipelineTaskConf]:
        if self.step_specs:
            if len(self.step_specs) != len(self.tasks):
                raise ValueError("Pipeline step specs and tasks differ in length.")
            return list(self.step_specs)
        used: Set[str] = set()
        steps: List[PipelineTaskConf] = []
        for task in self.tasks:
            candidate = task.name
            suffix = 2
            while candidate in used:
                candidate = f"{task.name}_{suffix}"
                suffix += 1
            used.add(candidate)
            steps.append(PipelineTaskConf(name=task.name, id=candidate))
        return steps

    def _check_declared_formats(self, source: Data, result: Optional[Data]) -> None:
        for decl in self.input_decls.values():
            if decl.role == "source" and _format_name(source.format) != decl.format:
                raise ValueError(
                    f"Source format '{_format_name(source.format)}' does not match declared '{decl.format}'."
                )
            if decl.role == "seed" and _format_name(self.seed.format) != decl.format:
                raise ValueError(
                    f"Seed format '{_format_name(self.seed.format)}' does not match declared '{decl.format}'."
                )
        if result is None:
            return
        for decl in self.output_decls.values():
            if _format_name(result.format) != decl.format:
                raise ValueError(
                    f"Result format '{_format_name(result.format)}' does not match declared '{decl.format}'."
                )

    def _bind_auto_inputs(
        self,
        task: KgTask,
        catalog: List[Data],
        ref_by_id: Dict[int, str],
    ) -> Tuple[List[Data], Dict[str, str]]:
        inputs: List[Data] = []
        input_refs: Dict[str, str] = {}
        for input_name, format_spec in task.input_spec.items():
            if input_name == "kg":
                chosen = catalog[-1]
                inputs.append(chosen)
                input_refs[input_name] = _ref_for(chosen, ref_by_id)
                continue
            for data_obj in catalog:
                if (
                    data_obj not in inputs
                    and _format_matches(data_obj, format_spec)
                    and len(inputs) < len(task.input_spec)
                ):
                    inputs.append(data_obj)
                    input_refs[input_name] = _ref_for(data_obj, ref_by_id)
                    break
        return inputs, input_refs

    def _bind_auto_outputs(
        self,
        *,
        task: KgTask,
        step_id: str,
        idx: int,
        result: Optional[Data],
        ref_by_id: Dict[int, str],
        gen_file_path: Callable[..., Path],
        task_hash: Optional[str],
    ) -> Tuple[List[Data], Dict[str, str]]:
        outputs: List[Data] = []
        output_refs: Dict[str, str] = {}
        for placeholder, format_spec in task.output_spec.items():
            if idx == len(self.tasks) - 1 and result:
                outputs.append(result)
                output_refs[placeholder] = _ref_for(result, ref_by_id)
                break
            output_path = gen_file_path(
                task=task,
                format_spec=format_spec,
                prefix=f"{idx}_",
                suffix=f"_{len(outputs)}",
                task_hash=task_hash,
            )
            output_data = Data(path=output_path, format=format_spec)
            ref = f"{step_id}.{placeholder}"
            ref_by_id[id(output_data)] = ref
            outputs.append(output_data)
            output_refs[placeholder] = ref
        return outputs, output_refs

    def _bind_explicit_step(
        self,
        *,
        task: KgTask,
        step: PipelineTaskConf,
        step_id: str,
        idx: int,
        result: Optional[Data],
        symbols: Dict[str, Data],
        ref_by_id: Dict[int, str],
        result_used: bool,
        gen_file_path: Callable[..., Path],
        task_hash: Optional[str],
    ) -> Tuple[List[Data], Dict[str, str], List[Data], Dict[str, str], bool]:
        if not step.inputs:
            raise ValueError(f"Step '{step_id}' is missing inputs under connect: explicit.")
        expected_inputs = set(task.input_spec)
        if set(step.inputs) != expected_inputs:
            raise ValueError(
                f"Step '{step_id}' inputs {sorted(step.inputs)} do not match "
                f"task ports {sorted(expected_inputs)}."
            )
        if step.outputs is not None and set(step.outputs) != set(task.output_spec):
            raise ValueError(
                f"Step '{step_id}' outputs {sorted(step.outputs)} do not match "
                f"task ports {sorted(task.output_spec)}."
            )

        inputs: List[Data] = []
        input_refs: Dict[str, str] = {}
        for input_name, format_spec in task.input_spec.items():
            ref = step.inputs[input_name]
            if ref not in symbols:
                raise ValueError(
                    f"Step '{step_id}' input '{input_name}' references unknown '{ref}'. "
                    f"Known: {sorted(symbols)}."
                )
            data_obj = symbols[ref]
            if not _format_matches(data_obj, format_spec):
                raise ValueError(
                    f"Step '{step_id}' input '{input_name}' expects {_format_name(format_spec)}, "
                    f"but '{ref}' is {_format_name(data_obj.format)}."
                )
            inputs.append(data_obj)
            input_refs[input_name] = ref

        outputs: List[Data] = []
        output_refs: Dict[str, str] = {}
        single_last_output = idx == len(self.tasks) - 1 and result is not None and len(task.output_spec) == 1
        for placeholder, format_spec in task.output_spec.items():
            if step.outputs is not None:
                ref = step.outputs[placeholder]
            elif single_last_output:
                ref = "$result"
            else:
                ref = f"{step_id}.{placeholder}"

            generated_ref = f"{step_id}.{placeholder}"
            if ref == "$result":
                if result is None:
                    raise ValueError(f"Step '{step_id}' binds '$result' but build() was given no result.")
                if idx != len(self.tasks) - 1:
                    raise ValueError(f"Step '{step_id}' cannot bind '$result'; it is not the last step.")
                if result_used:
                    raise ValueError("'$result' is already bound by another output.")
                if not _format_matches(result, format_spec):
                    raise ValueError(
                        f"Step '{step_id}' output '{placeholder}' expects {_format_name(format_spec)}, "
                        f"but '$result' is {_format_name(result.format)}."
                    )
                result_used = True
                output_data = result
            elif ref == generated_ref:
                if ref in symbols:
                    raise ValueError(f"Output ref '{ref}' is already defined.")
                output_path = gen_file_path(
                    task=task,
                    format_spec=format_spec,
                    prefix=f"{idx}_",
                    suffix=f"_{len(outputs)}",
                    task_hash=task_hash,
                )
                output_data = Data(path=output_path, format=format_spec)
                symbols[ref] = output_data
                ref_by_id[id(output_data)] = ref
            else:
                raise ValueError(
                    f"Step '{step_id}' output '{placeholder}' must be '{generated_ref}' or '$result'."
                )
            outputs.append(output_data)
            output_refs[placeholder] = ref
        return inputs, input_refs, outputs, output_refs, result_used

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

    def run(
        self,
        stable_files_override: bool = False,
        configCatalog: Optional[Mapping[str, ConfigurationProfile]] = None,
    ) -> List[KgTaskReport]:
        """Execute each task defined in the plan and collect the reports."""
        if not self.plan:
            raise ValueError("Pipeline plan is empty. Call build() first.")
        
        self.previous_was_skipped = True

        reports: List[KgTaskReport] = []
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

            configProfile = None
            profiles = configCatalog if configCatalog is not None else self.config_catalog
            if profiles:
                configProfile = profiles.get(task_spec.id) or profiles.get(task_spec.task)

            if self.previous_was_skipped:
                report = task.run(
                    task_spec.input,
                    task_spec.output,
                    stable_files_override=stable_files_override,
                    configProfile=configProfile,
                )
            else:
                report = task.run(
                    task_spec.input,
                    task_spec.output,
                    stable_files_override=True,
                    configProfile=configProfile,
                )

            if report.status != "skipped":
                self.previous_was_skipped = False

            reports.append(report)

        # pipeline_run_entity = reports_to_pipeline_run_entity(reports, self.name)
        # PipeKG.add_pipeline_run(pipeline_run_entity)

        return reports
    
    def __str__(self) -> str:
        return f"KgPipe(tasks={len(self.tasks)}, target={self.seed})"


# create dummy objects
# def dummy_function(inputs: dict[str, Data], outputs: dict[str, Data]):
#     print(inputs)
#     print(outputs)

# dummy_pipe = KgPipe(
#     tasks=[KgTask(name="dummy_task", input_spec={"source": DataFormat.RDF_NTRIPLES}, output_spec={"output": DataFormat.RDF_NTRIPLES}, function=dummy_function)],
#     seed=Data(path="dummy.ttl", format=DataFormat.RDF_NTRIPLES)
# )
# print("loaded dummy pipe")