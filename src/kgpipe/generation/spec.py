"""Versioned pipeline specification shared by YAML catalogs and ``KgPipe.build``.

A catalog entry is a user specification: task order, parameter values, and either
``connect: auto`` or explicit port bindings. ``KgPipe.build`` resolves that
specification into a ``KgPipePlan`` whose port refs are the DAG.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from yaml import Loader, SafeDumper, dump, load

from kgpipe.common.model.configuration import (
    ConfigurationDefinition,
    ConfigurationProfile,
    Parameter,
    ParameterBinding,
    ParameterType,
)
from kgpipe.common.model.data import DataFormat

PIPELINE_SPEC_SCHEMA = "kgpipe.pipeline/v1"

# Legacy MovieKG ``config`` keys. Values bind onto a task parameter when the
# normalized parameter name matches. Tasks that still read the environment keep
# doing so; this map does not set environment variables.
LEGACY_CONFIG_KEYS = {
    "ENTITY_MATCHING_THRESHOLD": "entity_matching_threshold",
    "RELATION_MATCHING_THRESHOLD": "relation_matching_threshold",
    "LLM_MODEL": "llm_model",
    "DEFAULT_LLM_MODEL_NAME": "llm_model",
}


def normalize_parameter_key(name: str) -> str:
    """Fold spaces, hyphens, and case so YAML keys match ``Parameter.name``."""
    return "_".join(name.strip().lower().replace("-", " ").split())


def legacy_parameter_key(key: str) -> str:
    if key in LEGACY_CONFIG_KEYS:
        return LEGACY_CONFIG_KEYS[key]
    return normalize_parameter_key(key)


class PipelinePortDecl(BaseModel):
    """A pipeline-level input or output port."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["source", "seed", "ontology", "result"]
    format: str

    @field_validator("format")
    @classmethod
    def _known_format(cls, value: str) -> str:
        try:
            DataFormat(value)
        except ValueError as exc:
            raise ValueError(f"Unknown format: {value}") from exc
        return value


class PipelineTaskConf(BaseModel):
    """One pipeline step. A bare task name in YAML is accepted as well."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    task: Optional[str] = None
    id: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    inputs: Optional[Dict[str, str]] = None
    outputs: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def _fill_task_name(self) -> "PipelineTaskConf":
        if self.name and self.task and self.name != self.task:
            raise ValueError(f"Step name '{self.name}' does not match task '{self.task}'.")
        if not self.name and self.task:
            self.name = self.task
        if not self.task and self.name:
            self.task = self.name
        if not self.name:
            raise ValueError("A pipeline step needs `name` or `task`.")
        return self

    @property
    def task_name(self) -> str:
        return self.name or self.task or ""


class PipelineConf(BaseModel):
    """One named pipeline in a catalog file."""

    model_config = ConfigDict(extra="forbid")

    description: str
    config: Optional[Dict[str, str]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    connect: Literal["auto", "explicit"] = "auto"
    inputs: Dict[str, PipelinePortDecl] = Field(default_factory=dict)
    output: Dict[str, PipelinePortDecl] = Field(default_factory=dict)
    tasks: List[Union[str, PipelineTaskConf]]

    @model_validator(mode="after")
    def _check_ports(self) -> "PipelineConf":
        for name, decl in self.inputs.items():
            if decl.role != name:
                raise ValueError(
                    f"Input port '{name}' must declare role '{name}'. "
                    "Ports are `$source`, `$seed`, and `$ontology`."
                )
            if decl.role == "result":
                raise ValueError("Declare the pipeline result under `output`, not `inputs`.")
        for name, decl in self.output.items():
            if name != "result" or decl.role != "result":
                raise ValueError("The only output port is `result` with role `result`.")
        return self


class PipelineConfs(BaseModel):
    """Catalog of named pipelines. The root `schema` key is not a pipeline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PIPELINE_SPEC_SCHEMA
    pipelines: Dict[str, PipelineConf]

    @property
    def root(self) -> Dict[str, PipelineConf]:
        """Mapping used by existing callers (`catalog.root[name]`)."""
        return self.pipelines


def load_pipeline_catalog(path: Path) -> PipelineConfs:
    with open(path, "r") as handle:
        raw = load(handle, Loader=Loader)
    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline catalog {path} must be a mapping of pipeline names.")
    schema_version, body = _split_catalog_schema(raw)
    return PipelineConfs(schema_version=schema_version, pipelines=body)


def _split_catalog_schema(raw: Mapping[str, Any]) -> tuple[str, Dict[str, Any]]:
    body = dict(raw)
    schema_version = PIPELINE_SPEC_SCHEMA
    declared = body.get("schema")
    if isinstance(declared, str):
        if declared != PIPELINE_SPEC_SCHEMA:
            raise ValueError(
                f"Unsupported pipeline catalog schema {declared!r}. "
                f"Supported: {PIPELINE_SPEC_SCHEMA}."
            )
        schema_version = declared
        body.pop("schema")
    return schema_version, body


def pipeline_task_name(task: Union[str, PipelineTaskConf]) -> str:
    if isinstance(task, str):
        return task
    return task.task_name


def pipeline_task_names(conf: PipelineConf) -> List[str]:
    return [pipeline_task_name(task) for task in conf.tasks]


def normalize_steps(conf: PipelineConf) -> List[PipelineTaskConf]:
    """Assign a stable step id and reject explicit bindings unless asked for."""
    for task in conf.tasks:
        if conf.connect != "explicit" and isinstance(task, PipelineTaskConf):
            if task.inputs or task.outputs:
                raise ValueError(
                    "Task input and output bindings require `connect: explicit`."
                )

    used: set[str] = set()
    steps: List[PipelineTaskConf] = []
    for task in conf.tasks:
        step = PipelineTaskConf(name=task) if isinstance(task, str) else task.model_copy(deep=True)
        if step.id:
            if step.id in used:
                raise ValueError(f"Duplicate step id '{step.id}'.")
            used.add(step.id)
        else:
            candidate = step.task_name
            suffix = 2
            while candidate in used:
                candidate = f"{step.task_name}_{suffix}"
                suffix += 1
            step.id = candidate
            used.add(candidate)
        steps.append(step)
    return steps


def _spec_index(spec: ConfigurationDefinition) -> Dict[str, Parameter]:
    index: Dict[str, Parameter] = {}
    for param in spec.parameters:
        index[param.name] = param
        index[normalize_parameter_key(param.name)] = param
        for native in param.native_keys:
            index[native] = param
            index[normalize_parameter_key(native)] = param
            if native in LEGACY_CONFIG_KEYS:
                index[LEGACY_CONFIG_KEYS[native]] = param
    return index


def _lookup_parameter(index: Mapping[str, Parameter], key: str) -> Optional[Parameter]:
    return index.get(key) or index.get(legacy_parameter_key(key))


def _coerce_parameter_value(param: Parameter, value: Any) -> Any:
    datatype = param.datatype
    if datatype == ParameterType.number:
        if isinstance(value, bool) or value is None:
            raise TypeError(f"Parameter '{param.name}' expects a number, got {value!r}.")
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return float(value)
        raise TypeError(f"Parameter '{param.name}' expects a number, got {type(value).__name__}.")
    if datatype == ParameterType.integer:
        if isinstance(value, bool):
            raise TypeError(f"Parameter '{param.name}' expects an integer, got {value!r}.")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value)
        raise TypeError(f"Parameter '{param.name}' expects an integer, got {type(value).__name__}.")
    if datatype == ParameterType.boolean:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise TypeError(f"Parameter '{param.name}' expects a boolean, got {value!r}.")
    if datatype == ParameterType.string:
        if not isinstance(value, str):
            raise TypeError(f"Parameter '{param.name}' expects a string, got {type(value).__name__}.")
        return value
    return value


def bind_config_catalog(
    conf: PipelineConf,
    tasks: Sequence[Any],
    steps: Sequence[PipelineTaskConf],
) -> Dict[str, ConfigurationProfile]:
    """Resolve parameter values for tasks that declare ``config_spec``.

    Binding order for each parameter is: declared default, legacy ``config``,
    pipeline ``parameters``, then the step's own ``parameters``.
    Profiles are keyed by step id. When the id is the task name, ``run`` still
    finds them.
    """
    if len(tasks) != len(steps):
        raise ValueError("Internal error: pipeline tasks and steps differ in length.")

    matched_pipeline_keys: set[str] = set()
    catalog: Dict[str, ConfigurationProfile] = {}

    for task, step in zip(tasks, steps):
        spec = getattr(task, "config_spec", None)
        step_id = step.id or step.task_name
        if spec is None:
            if step.parameters:
                raise ValueError(
                    f"Task '{step.task_name}' has parameter values but does not declare config_spec."
                )
            continue

        index = _spec_index(spec)
        by_name = {param.name: param for param in spec.parameters}
        values: Dict[str, Any] = {}
        for param in spec.parameters:
            if param.default_value is not None:
                values[param.name] = _coerce_parameter_value(param, param.default_value)

        for key, raw in (conf.config or {}).items():
            param = _lookup_parameter(index, key)
            if param is None:
                continue
            values[param.name] = _coerce_parameter_value(param, raw)

        for key, raw in conf.parameters.items():
            param = _lookup_parameter(index, key)
            if param is None:
                continue
            values[param.name] = _coerce_parameter_value(param, raw)
            matched_pipeline_keys.add(key)

        for key, raw in step.parameters.items():
            param = _lookup_parameter(index, key)
            if param is None:
                known = sorted(by_name)
                raise ValueError(
                    f"Unknown parameter '{key}' for task '{step.task_name}'. Known: {known}."
                )
            values[param.name] = _coerce_parameter_value(param, raw)

        profile = ConfigurationProfile(
            name=step_id,
            definition=spec,
            bindings=[
                ParameterBinding(parameter=by_name[name], value=value)
                for name, value in values.items()
            ],
        )
        task._validate_config(profile, spec)
        catalog[step_id] = profile

    unmatched = [key for key in conf.parameters if key not in matched_pipeline_keys]
    if unmatched:
        raise ValueError(
            "Pipeline parameters did not match any task config_spec: "
            + ", ".join(unmatched)
            + "."
        )
    return catalog


def profile_parameter_values(profile: Optional[ConfigurationProfile]) -> Dict[str, Any]:
    """Normalized parameter map stored on a resolved plan step."""
    if profile is None:
        return {}
    return {
        normalize_parameter_key(binding.parameter.name): binding.value
        for binding in profile.bindings
    }


def pipeline_conf_from_candidate(
    tasks: Sequence[Any],
    profiles: Mapping[str, ConfigurationProfile],
    *,
    description: str = "search candidate",
) -> PipelineConf:
    """User specification for one search candidate.

    ``tasks`` is the ordered list of live tasks. ``profiles`` is the candidate's
    configuration catalog, keyed by task name. The result is a
    ``kgpipe.pipeline/v1`` pipeline, the same object a YAML catalog entry loads.
    Parameter values are the profile bindings. ``connect`` stays ``auto``;
    ``KgPipe.build`` records the port refs.
    """
    steps: List[Union[str, PipelineTaskConf]] = []
    for task in tasks:
        profile = profiles.get(getattr(task, "name", ""))
        if profile is None or not profile.bindings:
            steps.append(task.name)
            continue
        steps.append(PipelineTaskConf(
            name=task.name,
            parameters={
                normalize_parameter_key(binding.parameter.name): binding.value
                for binding in profile.bindings
            },
        ))
    return PipelineConf(description=description, connect="auto", tasks=steps)


def pipeline_conf_from_plan(plan: Any, *, description: str) -> PipelineConf:
    """Write a resolved plan back as ``connect: explicit``.

    Rebuilding that specification produces the same port refs, parameter
    values, and fingerprint. Paths stay out of the specification.
    """
    steps = [
        PipelineTaskConf(
            id=step.id or step.task,
            name=step.task,
            inputs=dict(step.inputs),
            outputs=dict(step.outputs),
            parameters=dict(step.parameters),
        )
        for step in plan.steps
    ]
    return PipelineConf(description=description, connect="explicit", tasks=steps)


def dump_pipeline_catalog(pipelines: Mapping[str, PipelineConf]) -> str:
    """Serialize a catalog, including the ``schema`` version key."""
    body: Dict[str, Any] = {"schema": PIPELINE_SPEC_SCHEMA}
    for name, conf in pipelines.items():
        body[name] = conf.model_dump(mode="json", exclude_none=True)
    text = dump(body, Dumper=SafeDumper, sort_keys=False)
    if not text.endswith("\n"):
        text += "\n"
    return text


def candidate_fingerprint(steps: Sequence[Mapping[str, Any]]) -> str:
    """Hash task identity, port refs, and resolved parameter values.

    File paths are execution details and are not part of the candidate.
    """
    payload = {"schema": PIPELINE_SPEC_SCHEMA, "steps": list(steps)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
