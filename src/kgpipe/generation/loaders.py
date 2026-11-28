"""
YAML pipeline loaders and template utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from pydantic import BaseModel

from kgpipe.common import Data, KgPipe, Registry
from kgpipe.common.models import DataFormat
from kgpipe.generation.spec import (
    CatalogValidationError,
    PipelineConf,
    PipelineConfs,
    PipelineTaskConf,
    bind_config_catalog,
    load_pipeline_catalog,
    normalize_steps,
    pipeline_conf_from_candidate,
    pipeline_task_name,
    pipeline_task_names,
)

__all__ = [
    "CatalogValidationError",
    "PipelineCatalog",
    "PipelineConf",
    "PipelineConfs",
    "PipelineDefinition",
    "bind_config_catalog",
    "build_candidate",
    "build_from_conf",
    "build_from_yaml",
    "get_test_data",
    "load_pipeline_catalog",
    "normalize_steps",
    "pipeline_conf_from_candidate",
    "pipeline_task_name",
    "pipeline_task_names",
    "select_pipeline_conf",
    "ssp_pipeline",
    "validate_pipeline",
]

class PipelineDefinition(BaseModel):
    name: str
    description: str
    pipeline: str
    source: str
    target: str
    source_format: str
    target_format: str
    source_schema: str
    target_schema: str

class PipelineCatalog(BaseModel):
    name: str
    description: str
    pipelines: list[PipelineDefinition]


catalog = PipelineCatalog(
    name="rdf_pipelines",
    description="RDF pipelines",
    pipelines=[]
)

def get_test_data(file_name: str) -> Path:
    return Path(__file__).parent / "test_data" / file_name


def ssp_pipeline(name: str, tasks: list, target_data: Data, data_dir: str) -> KgPipe:
    pipe = KgPipe(tasks, target_data, data_dir, name=name)
    return pipe


def build_from_conf(name: str, conf: PipelineConf, target_data: Data, data_dir: str, tasks=None) -> KgPipe:
    """Build a pipeline from a catalog entry.

    ``tasks`` is the ordered list of live task objects. Search candidates pass
    the tasks they already hold. Without them, each step name is resolved
    through the registry.
    """
    resolved, steps = resolve_pipeline_tasks(name, conf, tasks=tasks)
    try:
        config_catalog = bind_config_catalog(conf, resolved, steps)
    except (CatalogValidationError, ValueError, TypeError) as exc:
        raise CatalogValidationError(f"Pipeline '{name}': {exc}") from exc
    pipe = ssp_pipeline(name, resolved, target_data, data_dir)
    pipe.connect = conf.connect
    pipe.step_specs = steps
    pipe.config_catalog = config_catalog
    pipe.input_decls = dict(conf.inputs)
    pipe.output_decls = dict(conf.output)
    return pipe


def resolve_pipeline_tasks(
    name: str,
    conf: PipelineConf,
    tasks: Optional[Sequence[Any]] = None,
) -> tuple[list, list[PipelineTaskConf]]:
    """Resolve registry tasks and reject unknown names or backend mismatches."""
    try:
        steps = normalize_steps(conf)
    except ValueError as exc:
        raise CatalogValidationError(f"Pipeline '{name}': {exc}") from exc
    if not steps:
        raise CatalogValidationError(
            f"Pipeline '{name}' has no tasks. Add at least one task name or step."
        )
    if tasks is None:
        resolved = []
        for step in steps:
            try:
                resolved.append(Registry.get_task(step.task_name))
            except KeyError as exc:
                raise CatalogValidationError(
                    f"Unknown task '{step.task_name}' in pipeline '{name}'. "
                    "Import the task package or run `kgpipe discover` first."
                ) from exc
    else:
        resolved = list(tasks)
        if len(resolved) != len(steps):
            raise CatalogValidationError(
                f"Pipeline '{name}' has {len(steps)} steps and {len(resolved)} tasks."
            )
        for task, step in zip(resolved, steps):
            if task.name != step.task_name:
                raise CatalogValidationError(
                    f"Task '{task.name}' does not match spec step '{step.task_name}'."
                )
    _validate_backends(name, conf, resolved, steps)
    return resolved, steps


def _validate_backends(
    name: str,
    conf: PipelineConf,
    tasks: Sequence[Any],
    steps: Sequence[PipelineTaskConf],
) -> None:
    pipeline_backend = conf.backend
    for task, step in zip(tasks, steps):
        requested = step.backend or pipeline_backend
        if requested is None:
            continue
        declared = getattr(task, "backend", None) or "python"
        if str(declared).strip().lower() != requested:
            raise CatalogValidationError(
                f"Pipeline '{name}' step '{step.id or step.task_name}' requests "
                f"backend {requested!r} but task '{task.name}' is registered "
                f"with backend {declared!r}."
            )


def build_candidate(
    name: str,
    tasks,
    profiles,
    target_data: Data,
    data_dir: str,
    *,
    description: str = "search candidate",
):
    """Turn a search candidate into a pipeline built from the shared spec."""
    conf = pipeline_conf_from_candidate(tasks, profiles, description=description)
    pipe = build_from_conf(name, conf, target_data, data_dir, tasks=tasks)
    return pipe, conf


def select_pipeline_conf(
    catalog: PipelineConfs,
    name: Optional[str] = None,
) -> tuple[str, PipelineConf]:
    """Pick one catalog entry. A single-pipeline file does not need a name."""
    if name is not None:
        if name not in catalog.root:
            available = ", ".join(sorted(catalog.root)) or "(none)"
            raise CatalogValidationError(
                f"Unknown pipeline {name!r}. Available: {available}."
            )
        return name, catalog.root[name]
    if len(catalog.root) == 1:
        only = next(iter(catalog.root))
        return only, catalog.root[only]
    available = ", ".join(sorted(catalog.root)) or "(none)"
    location = catalog.source_path or "catalog"
    raise CatalogValidationError(
        f"{location} defines multiple pipelines ({available}). Pass name= to select one."
    )


def build_from_yaml(
    yaml_path: Path,
    name: Optional[str] = None,
    *,
    source: Optional[Union[str, Path]] = None,
    seed: Optional[Union[str, Path]] = None,
    target: Optional[Union[str, Path]] = None,
    result: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    ontology: Optional[Union[str, Path]] = None,
    tasks=None,
    environ: Optional[Mapping[str, str]] = None,
    stable_files: bool = True,
) -> KgPipe:
    """Load a catalog entry, bind files, and build a plan.

    Path precedence, highest first: function arguments, pipeline ``defaults``,
    catalog ``defaults``. Relative paths are resolved against the YAML file
    directory. ``Pipeline.from_yaml`` is the public wrapper around this loader.
    """
    loaded = load_pipeline_catalog(Path(yaml_path), environ=environ)
    pipeline_name, conf = select_pipeline_conf(loaded, name)
    resolved, _steps = resolve_pipeline_tasks(pipeline_name, conf, tasks=tasks)
    paths = _resolve_run_paths(
        loaded,
        pipeline_name,
        source=source,
        seed=seed,
        target=target,
        result=result,
        data_dir=data_dir,
        ontology=ontology,
    )
    source_data = Data(
        path=paths["source"],
        format=_format_for_role(conf, "source", resolved, paths["source"]),
    )
    seed_data = Data(
        path=paths["seed"],
        format=_format_for_role(conf, "seed", resolved, paths["seed"]),
    )
    result_data = Data(
        path=paths["result"],
        format=_format_for_role(conf, "result", resolved, paths["result"]),
    )
    try:
        pipe = build_from_conf(
            pipeline_name,
            conf,
            seed_data,
            str(paths["data_dir"]),
            tasks=resolved,
        )
        if paths["ontology"] is not None:
            pipe.ontology_path = str(paths["ontology"])
        pipe.build(source=source_data, result=result_data, stable_files=stable_files)
    except CatalogValidationError:
        raise
    except ValueError as exc:
        raise CatalogValidationError(f"Pipeline '{pipeline_name}': {exc}") from exc
    return pipe


def validate_pipeline(
    yaml_path: Path,
    name: Optional[str] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    tasks=None,
) -> KgPipe:
    """Load, bind, and plan a catalog entry without executing it.

    Invalid task combinations, parameter types, backends, and port wiring fail
    here rather than during ``run()``.
    """
    return build_from_yaml(yaml_path, name, environ=environ, tasks=tasks)


def _resolve_run_paths(
    loaded: PipelineConfs,
    name: str,
    *,
    source: Optional[Union[str, Path]],
    seed: Optional[Union[str, Path]],
    target: Optional[Union[str, Path]],
    result: Optional[Union[str, Path]],
    data_dir: Optional[Union[str, Path]],
    ontology: Optional[Union[str, Path]],
) -> dict[str, Optional[Path]]:
    base_dir = loaded.source_path.parent if loaded.source_path is not None else Path.cwd()
    defaults = loaded.resolved_defaults(name, base_dir=base_dir)
    seed_arg = seed if seed is not None else target

    def _pick(explicit: Optional[Union[str, Path]], default: Optional[str], fallback: Path) -> Path:
        if explicit is not None:
            return _as_path(explicit, base_dir=base_dir)
        if default:
            return Path(default)
        return fallback

    work = _pick(data_dir, defaults.data_dir, base_dir / ".kgpipe" / name)
    source_path = _pick(source, defaults.source, work / "source")
    seed_path = _pick(seed_arg, defaults.seed, work / "seed")
    result_path = _pick(result, defaults.result, work / "result.nt")
    ontology_path = None
    if ontology is not None:
        ontology_path = _as_path(ontology, base_dir=base_dir)
    elif defaults.ontology:
        ontology_path = Path(defaults.ontology)
    return {
        "data_dir": work,
        "source": source_path,
        "seed": seed_path,
        "result": result_path,
        "ontology": ontology_path,
    }


def _as_path(value: Union[str, Path], *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _format_for_role(
    conf: PipelineConf,
    role: str,
    tasks: Sequence[Any],
    path: Path,
) -> DataFormat:
    if role == "result":
        for decl in conf.output.values():
            return DataFormat(decl.format)
        if tasks:
            outputs = list(tasks[-1].output_spec.values())
            if outputs:
                return outputs[0]
    else:
        if role in conf.inputs:
            return DataFormat(conf.inputs[role].format)
        if tasks:
            if role == "seed" and "kg" in tasks[0].input_spec:
                return tasks[0].input_spec["kg"]
            for port, fmt in tasks[0].input_spec.items():
                if port != "kg":
                    return fmt
    suffix = path.suffix.lower().lstrip(".")
    if suffix:
        try:
            return DataFormat.from_extension(suffix)
        except ValueError:
            pass
    return DataFormat.RDF_NTRIPLES
