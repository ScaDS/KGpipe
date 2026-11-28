"""Versioned pipeline specification shared by YAML catalogs and ``KgPipe.build``.

A catalog entry is a user specification: task order, parameter values, and either
``connect: auto`` or explicit port bindings. ``KgPipe.build`` resolves that
specification into a ``KgPipePlan`` whose port refs are the DAG.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from yaml import SafeDumper, SafeLoader, YAMLError, dump, load

from kgpipe.common.model.configuration import (
    ConfigurationDefinition,
    ConfigurationProfile,
    Parameter,
    ParameterBinding,
    ParameterType,
)
from kgpipe.common.model.data import DataFormat

PIPELINE_SPEC_SCHEMA = "kgpipe.pipeline/v1"
KNOWN_BACKENDS = ("python", "docker", "remote")
RESERVED_CATALOG_KEYS = frozenset({"schema", "defaults", "env"})
ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
DEFAULT_PATH_FIELDS = ("data_dir", "source", "seed", "target", "result", "ontology")

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


class CatalogValidationError(ValueError):
    """A pipeline catalog failed to load or could not be executed as written."""


def interpolate_env(value: Any, environ: Mapping[str, str]) -> Any:
    """Replace ``${VAR}`` and ``${VAR:-default}`` in catalog strings.

    Bare ``$source`` / ``$seed`` / ``$result`` port refs are left untouched.
    A missing variable without a default is an error.
    """

    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            if name in environ:
                return environ[name]
            if default is not None:
                return default
            raise CatalogValidationError(
                f"Environment variable '{name}' is not set and has no default. "
                f"Set {name} or write ${{{name}:-default}}."
            )

        return ENV_PLACEHOLDER.sub(_replace, value)
    if isinstance(value, list):
        return [interpolate_env(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: interpolate_env(item, environ) for key, item in value.items()}
    return value


def resolve_catalog_path(value: str, *, base_dir: Path) -> Path:
    """Resolve a catalog path against the YAML file directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _normalize_backend(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in KNOWN_BACKENDS:
        raise ValueError(
            f"Unknown backend {value!r}. Supported: {', '.join(KNOWN_BACKENDS)}."
        )
    return normalized


class PipelineDefaults(BaseModel):
    """Execution paths and files. Not part of the candidate fingerprint."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Optional[str] = None
    source: Optional[str] = None
    seed: Optional[str] = None
    target: Optional[str] = None
    result: Optional[str] = None
    ontology: Optional[str] = None

    @model_validator(mode="after")
    def _alias_target(self) -> "PipelineDefaults":
        if self.seed is None and self.target is not None:
            self.seed = self.target
        return self

    def merged_with(self, override: Optional["PipelineDefaults"]) -> "PipelineDefaults":
        if override is None:
            return self
        data = self.model_dump(exclude_none=True)
        data.update(override.model_dump(exclude_none=True))
        return PipelineDefaults.model_validate(data)

    def resolve(self, *, base_dir: Path) -> "PipelineDefaults":
        values: Dict[str, Optional[str]] = {}
        for field_name in DEFAULT_PATH_FIELDS:
            raw = getattr(self, field_name)
            if raw is None:
                values[field_name] = None
            else:
                values[field_name] = str(resolve_catalog_path(raw, base_dir=base_dir))
        return PipelineDefaults.model_validate(values)


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
    backend: Optional[str] = None
    backend_options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def _known_backend(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_backend(value)

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
    defaults: Optional[PipelineDefaults] = None
    backend: Optional[str] = None
    backend_options: Dict[str, Any] = Field(default_factory=dict)
    tasks: List[Union[str, PipelineTaskConf]]

    @field_validator("backend")
    @classmethod
    def _known_backend(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_backend(value)

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
    defaults: Optional[PipelineDefaults] = None
    env: Dict[str, str] = Field(default_factory=dict)
    pipelines: Dict[str, PipelineConf]
    source_path: Optional[Path] = None

    @property
    def root(self) -> Dict[str, PipelineConf]:
        """Mapping used by existing callers (`catalog.root[name]`)."""
        return self.pipelines

    def resolved_defaults(self, name: str, *, base_dir: Optional[Path] = None) -> PipelineDefaults:
        """Catalog defaults overlaid with the named pipeline, paths resolved."""
        if name not in self.pipelines:
            available = ", ".join(sorted(self.pipelines)) or "(none)"
            raise CatalogValidationError(
                f"Unknown pipeline {name!r}. Available: {available}."
            )
        merged = (self.defaults or PipelineDefaults()).merged_with(self.pipelines[name].defaults)
        root = base_dir
        if root is None and self.source_path is not None:
            root = self.source_path.parent
        if root is None:
            return merged
        return merged.resolve(base_dir=root)


def load_pipeline_catalog(
    path: Path,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> PipelineConfs:
    """Load a ``kgpipe.pipeline/v1`` catalog from YAML.

    Relative paths in ``defaults`` stay as written. ``build_from_yaml`` and
    ``PipelineConfs.resolved_defaults`` resolve them against the file directory.
    ``${VAR}`` / ``${VAR:-default}`` are expanded using ``environ`` (or
    ``os.environ``) after the catalog ``env`` map.
    """
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise CatalogValidationError(f"Pipeline catalog not found: {catalog_path}")
    try:
        with open(catalog_path, "r", encoding="utf-8") as handle:
            raw = load(handle, Loader=SafeLoader)
    except YAMLError as exc:
        raise CatalogValidationError(f"Invalid YAML in {catalog_path}: {exc}") from exc
    return load_pipeline_catalog_data(raw, path=catalog_path, environ=environ)


def load_pipeline_catalog_data(
    raw: Any,
    *,
    path: Path,
    environ: Optional[Mapping[str, str]] = None,
) -> PipelineConfs:
    if not isinstance(raw, dict):
        raise CatalogValidationError(
            f"Pipeline catalog {path} must be a mapping of pipeline names."
        )
    try:
        schema_version, defaults, catalog_env, body = _split_catalog_header(raw)
        merged_env = {str(key): str(value) for key, value in catalog_env.items()}
        if environ is None:
            merged_env.update(os.environ)
        else:
            merged_env.update({str(key): str(value) for key, value in environ.items()})
        body = interpolate_env(body, merged_env)
        defaults_raw = interpolate_env(defaults, merged_env) if defaults is not None else None
        pipelines = _parse_pipelines(body, path=path)
        parsed_defaults = (
            PipelineDefaults.model_validate(defaults_raw) if defaults_raw is not None else None
        )
    except CatalogValidationError:
        raise
    except ValidationError as exc:
        raise CatalogValidationError(_format_catalog_validation(path, exc)) from exc
    return PipelineConfs(
        schema_version=schema_version,
        defaults=parsed_defaults,
        env=catalog_env,
        pipelines=pipelines,
        source_path=path,
    )


def _split_catalog_header(
    raw: Mapping[str, Any],
) -> tuple[str, Optional[Any], Dict[str, str], Dict[str, Any]]:
    body = dict(raw)
    schema_version = PIPELINE_SPEC_SCHEMA
    declared = body.pop("schema", None)
    if declared is not None:
        if not isinstance(declared, str) or declared != PIPELINE_SPEC_SCHEMA:
            raise CatalogValidationError(
                f"Unsupported pipeline catalog schema {declared!r}. "
                f"Supported: {PIPELINE_SPEC_SCHEMA}."
            )
        schema_version = declared
    defaults = body.pop("defaults", None)
    if defaults is not None and not isinstance(defaults, dict):
        raise CatalogValidationError("`defaults` must be a mapping of path keys.")
    catalog_env_raw = body.pop("env", None) or {}
    if not isinstance(catalog_env_raw, dict):
        raise CatalogValidationError("`env` must be a mapping of environment names to values.")
    catalog_env = {str(key): str(value) for key, value in catalog_env_raw.items()}
    return schema_version, defaults, catalog_env, body


def _parse_pipelines(body: Mapping[str, Any], *, path: Path) -> Dict[str, PipelineConf]:
    if not body:
        raise CatalogValidationError(f"Pipeline catalog {path} does not define any pipelines.")
    pipelines: Dict[str, PipelineConf] = {}
    errors: List[str] = []
    for name, raw_conf in body.items():
        if name in RESERVED_CATALOG_KEYS:
            continue
        if not isinstance(raw_conf, dict):
            errors.append(
                f"pipeline '{name}': expected a mapping with description and tasks, "
                f"got {type(raw_conf).__name__}."
            )
            continue
        try:
            pipelines[name] = PipelineConf.model_validate(raw_conf)
        except ValidationError as exc:
            errors.append(_format_pipeline_validation(name, exc))
    if errors:
        raise CatalogValidationError(
            f"Invalid pipeline catalog {path}:\n" + "\n".join(f"  - {item}" for item in errors)
        )
    return pipelines


def _format_catalog_validation(path: Path, exc: ValidationError) -> str:
    return f"Invalid pipeline catalog {path}: {_format_validation_errors(exc)}"


def _format_pipeline_validation(name: str, exc: ValidationError) -> str:
    return f"pipeline '{name}': {_format_validation_errors(exc)}"


def _format_validation_errors(exc: ValidationError) -> str:
    parts: List[str] = []
    for err in exc.errors():
        loc = [str(item) for item in err.get("loc", ()) if item != "pipelines"]
        where = ".".join(loc)
        message = err.get("msg", "invalid value")
        err_type = err.get("type", "")
        if err_type == "extra_forbidden":
            field = loc[-1] if loc else "unknown"
            parts.append(f"unexpected field '{field}'")
            continue
        if where:
            parts.append(f"{where}: {message}")
        else:
            parts.append(message)
    return "; ".join(parts) or "invalid catalog"


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
        if step.backend is None and conf.backend is not None:
            step.backend = conf.backend
        if not step.backend_options and conf.backend_options:
            step.backend_options = dict(conf.backend_options)
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
    try:
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
    except (TypeError, ValueError) as exc:
        raise CatalogValidationError(str(exc)) from exc
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
                raise CatalogValidationError(
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
                raise CatalogValidationError(
                    f"Unknown parameter '{key}' for task '{step.task_name}'. "
                    f"Known: {known or '(none)'}."
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
        try:
            task._validate_config(profile, spec)
        except (TypeError, ValueError) as exc:
            raise CatalogValidationError(
                f"Invalid parameters for task '{step.task_name}': {exc}"
            ) from exc
        catalog[step_id] = profile

    unmatched = [key for key in conf.parameters if key not in matched_pipeline_keys]
    if unmatched:
        raise CatalogValidationError(
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


def dump_pipeline_catalog(
    pipelines: Union[Mapping[str, PipelineConf], PipelineConfs],
    *,
    defaults: Optional[PipelineDefaults] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Serialize a catalog, including the ``schema`` version key."""
    catalog_defaults = defaults
    catalog_env: Mapping[str, str] = env or {}
    if isinstance(pipelines, PipelineConfs):
        if catalog_defaults is None:
            catalog_defaults = pipelines.defaults
        if env is None:
            catalog_env = pipelines.env
        pipelines = pipelines.pipelines
    body: Dict[str, Any] = {"schema": PIPELINE_SPEC_SCHEMA}
    if catalog_env:
        body["env"] = dict(catalog_env)
    if catalog_defaults is not None:
        dumped_defaults = catalog_defaults.model_dump(mode="json", exclude_none=True)
        if dumped_defaults:
            body["defaults"] = dumped_defaults
    for name, conf in pipelines.items():
        body[name] = _dump_pipeline_conf(conf)
    text = dump(body, Dumper=SafeDumper, sort_keys=False)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _dump_pipeline_conf(conf: PipelineConf) -> Dict[str, Any]:
    dumped = conf.model_dump(mode="json", exclude_none=True)
    for key in ("backend_options", "defaults"):
        if not dumped.get(key):
            dumped.pop(key, None)
    steps = []
    for task in dumped.get("tasks", []):
        if isinstance(task, dict) and not task.get("backend_options"):
            task.pop("backend_options", None)
        steps.append(task)
    dumped["tasks"] = steps
    return dumped


def candidate_fingerprint(steps: Sequence[Mapping[str, Any]]) -> str:
    """Hash task identity, port refs, and resolved parameter values.

    File paths are execution details and are not part of the candidate.
    """
    payload = {"schema": PIPELINE_SPEC_SCHEMA, "steps": list(steps)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
