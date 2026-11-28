"""Public Python API for fixed pipelines.

A :class:`Pipeline` is a typed ``kgpipe.pipeline/v1`` specification plus the
files it reads and writes. ``plan()`` resolves that specification into a
``KgPipePlan``. ``run()`` executes the plan and returns a :class:`PipelineRun`.

YAML catalogs, ``Pipeline(...)``, and ``Pipeline.from_conf`` all go through
``PipelineConf`` and ``KgPipe.build``, so equivalent definitions share one
fingerprint. ``KgPipe`` remains the execution object.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, ValidationError

from kgpipe.common.model.data import Data, DataFormat
from kgpipe.common.model.pipeline import KgPipe, KgPipePlan
from kgpipe.common.model.task import KgTask, KgTaskReport
from kgpipe.common.registry import Registry
from kgpipe.generation.loaders import (
    _as_path,
    _format_for_role,
    build_from_conf,
    build_from_yaml,
    resolve_pipeline_tasks,
    select_pipeline_conf,
)
from kgpipe.generation.spec import (
    CatalogValidationError,
    PipelineConf,
    PipelineDefaults,
    PipelinePortDecl,
    PipelineTaskConf,
    load_pipeline_catalog,
    normalize_steps,
)

FileRef = Union[str, Path, Data]
TaskRef = Union[str, PipelineTaskConf, KgTask, Mapping[str, Any]]

__all__ = [
    "CatalogValidationError",
    "KgPipePlan",
    "Pipeline",
    "PipelineConf",
    "PipelineRun",
    "PipelineTaskConf",
]


class PipelineRun(BaseModel):
    """Structured result of :meth:`Pipeline.run`.

    ``status`` is ``success`` when at least one task succeeded and none failed,
    ``skipped`` when every task was skipped, and ``failed`` when a task failed
    or execution stopped. Task failures stay on ``reports``. ``duration`` is
    the wall time of the execution call. ``plan`` is the plan that was executed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    status: Literal["success", "failed", "skipped"]
    plan: KgPipePlan
    reports: list[KgTaskReport]
    result: Optional[Data] = None
    error: Optional[str] = None
    duration: float = 0.0


class Pipeline:
    """A fixed pipeline defined in Python or loaded from a catalog.

    ``tasks`` entries are registry names, step mappings, :class:`PipelineTaskConf`
    objects, or live :class:`KgTask` objects. Paths are strings, :class:`Path`
    objects, or :class:`Data`. Relative paths resolve against ``base_dir``,
    which defaults to the current directory. ``seed`` and ``target`` are the
    same input.
    """

    def __init__(
        self,
        tasks: Union[TaskRef, Sequence[TaskRef]],
        *,
        name: str = "pipeline",
        description: str = "",
        parameters: Optional[Mapping[str, Any]] = None,
        connect: Literal["auto", "explicit"] = "auto",
        inputs: Optional[Mapping[str, Any]] = None,
        output: Optional[Mapping[str, Any]] = None,
        backend: Optional[str] = None,
        backend_options: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        defaults: Optional[Union[PipelineDefaults, Mapping[str, Any]]] = None,
        source: Optional[FileRef] = None,
        seed: Optional[FileRef] = None,
        target: Optional[FileRef] = None,
        result: Optional[FileRef] = None,
        data_dir: Optional[Union[str, Path]] = None,
        ontology: Optional[FileRef] = None,
        base_dir: Optional[Union[str, Path]] = None,
        stable_files: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self._task_items = _as_task_list(name, tasks)
        self._parameters = dict(parameters or {})
        self._connect = connect
        self._inputs = dict(inputs or {})
        self._output = dict(output or {})
        self._backend = backend
        self._backend_options = dict(backend_options or {})
        self._config = None if config is None else dict(config)
        self._defaults = defaults
        self._source = source
        self._seed = seed
        self._target = target
        self._result = result
        self._data_dir = data_dir
        self._ontology = ontology
        self._base_dir = Path(base_dir) if base_dir is not None else None
        self.stable_files = stable_files
        self._conf: Optional[PipelineConf] = None
        self._live: Optional[list[KgTask]] = None
        self._pipe: Optional[KgPipe] = None

    @classmethod
    def from_conf(
        cls,
        name: str,
        conf: PipelineConf,
        *,
        tasks: Optional[Sequence[KgTask]] = None,
        source: Optional[FileRef] = None,
        seed: Optional[FileRef] = None,
        target: Optional[FileRef] = None,
        result: Optional[FileRef] = None,
        data_dir: Optional[Union[str, Path]] = None,
        ontology: Optional[FileRef] = None,
        base_dir: Optional[Union[str, Path]] = None,
        stable_files: bool = True,
    ) -> "Pipeline":
        """Build from a :class:`PipelineConf`.

        ``tasks`` is the live task list when those objects are already in
        hand. Without it, each step name is resolved through the registry.
        """
        pipeline = cls(
            list(conf.tasks) if tasks is None else list(tasks),
            name=name,
            description=conf.description,
            source=source,
            seed=seed,
            target=target,
            result=result,
            data_dir=data_dir,
            ontology=ontology,
            base_dir=base_dir,
            stable_files=stable_files,
        )
        pipeline._conf = conf
        if tasks is not None:
            pipeline._live = list(tasks)
        return pipeline

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        *,
        source: Optional[Union[str, Path]] = None,
        seed: Optional[Union[str, Path]] = None,
        target: Optional[Union[str, Path]] = None,
        result: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        ontology: Optional[Union[str, Path]] = None,
        tasks: Optional[Sequence[KgTask]] = None,
        environ: Optional[Mapping[str, str]] = None,
        stable_files: bool = True,
    ) -> "Pipeline":
        """Load one catalog entry, plan it, and wrap the result.

        Path precedence matches ``build_from_yaml``: arguments, then the
        pipeline ``defaults``, then the catalog ``defaults``. A file with one
        pipeline does not need ``name``.
        """
        catalog_path = Path(path)
        loaded = load_pipeline_catalog(catalog_path, environ=environ)
        pipeline_name, conf = select_pipeline_conf(loaded, name)
        pipe = build_from_yaml(
            catalog_path,
            pipeline_name,
            source=source,
            seed=seed,
            target=target,
            result=result,
            data_dir=data_dir,
            ontology=ontology,
            tasks=tasks,
            environ=environ,
            stable_files=stable_files,
        )
        pipeline = cls(list(conf.tasks), name=pipeline_name, description=conf.description)
        pipeline._conf = conf
        pipeline._pipe = pipe
        if tasks is not None:
            pipeline._live = list(tasks)
        return pipeline

    @classmethod
    def from_kgpipe(
        cls,
        pipe: KgPipe,
        *,
        source: Optional[FileRef] = None,
        result: Optional[FileRef] = None,
        stable_files: bool = True,
    ) -> "Pipeline":
        """Wrap a ``KgPipe``.

        An existing plan is kept, including its fingerprint. ``source`` and
        ``result`` are the ``KgPipe.build`` arguments and are used only when
        that plan is still empty. The call builds on ``pipe`` itself.
        """
        if pipe.plan.steps:
            if source is not None or result is not None:
                raise CatalogValidationError(
                    f"Pipeline '{pipe.name}' already has a plan. "
                    "source and result apply only before KgPipe.build()."
                )
        else:
            if source is None:
                raise CatalogValidationError(
                    f"Pipeline '{pipe.name}' has no plan. Pass source= "
                    "or call KgPipe.build() before Pipeline.from_kgpipe()."
                )
            source_data = _coerce_build_data(source, pipe, role="source")
            result_data = None if result is None else _coerce_build_data(result, pipe, role="result")
            try:
                pipe.build(source=source_data, result=result_data, stable_files=stable_files)
            except CatalogValidationError:
                raise
            except ValueError as exc:
                raise CatalogValidationError(f"Pipeline '{pipe.name}': {exc}") from exc
        pipeline = cls(
            list(pipe.tasks),
            name=pipe.name or "pipeline",
            description=pipe.name or "pipeline",
        )
        pipeline._conf = _conf_from_kgpipe(pipe)
        pipeline._live = list(pipe.tasks)
        pipeline._pipe = pipe
        return pipeline

    @property
    def conf(self) -> PipelineConf:
        """Typed specification. Paths and file names are not part of it."""
        if self._conf is None:
            self._conf = self._build_conf()
        return self._conf

    def validate(self) -> KgPipePlan:
        """Resolve ports and parameters. Does not execute tasks."""
        return self._ensure().plan

    def plan(self) -> KgPipePlan:
        """Resolved execution plan. Same object as ``validate()``."""
        return self.validate()

    def run(self, *, stable_files_override: bool = False) -> PipelineRun:
        """Plan the pipeline, execute it, and return a :class:`PipelineRun`.

        An invalid specification raises :class:`CatalogValidationError` before
        any task runs. A task that returns ``failed``, or an execution error
        such as a missing input file, is ``status='failed'`` on the result.
        """
        pipe = self._ensure()
        started = time.perf_counter()
        try:
            reports = list(pipe.run(stable_files_override=stable_files_override))
            error = None
        except Exception as exc:
            reports = []
            error = f"{type(exc).__name__}: {exc}"
        duration = time.perf_counter() - started
        status = _run_status(reports)
        if error is None and status == "failed":
            error = next((report.error for report in reports if report.status == "failed" and report.error), None)
            if error is None and not reports:
                error = "Pipeline produced no task reports."
        return PipelineRun(
            name=self.name,
            status=status,
            plan=pipe.plan,
            reports=reports,
            result=pipe.plan.result,
            error=error,
            duration=duration,
        )

    def to_kgpipe(self) -> KgPipe:
        """Execution object compiled from this specification."""
        return self._ensure()

    def _build_conf(self) -> PipelineConf:
        tasks: list[Union[str, PipelineTaskConf]] = []
        for item in self._task_items:
            if isinstance(item, KgTask):
                tasks.append(item.name)
            elif isinstance(item, str):
                tasks.append(item)
            elif isinstance(item, PipelineTaskConf):
                tasks.append(item)
            elif isinstance(item, Mapping):
                try:
                    tasks.append(PipelineTaskConf.model_validate(dict(item)))
                except ValidationError as exc:
                    raise _conf_error(self.name, exc) from exc
            else:
                raise CatalogValidationError(
                    f"Pipeline '{self.name}': task must be a name, PipelineTaskConf, "
                    f"mapping, or KgTask, got {type(item).__name__}."
                )
        try:
            return PipelineConf(
                description=self.description or self.name,
                parameters=dict(self._parameters),
                connect=self._connect,
                inputs={key: _port(key, value) for key, value in self._inputs.items()},
                output={key: _port(key, value) for key, value in self._output.items()},
                backend=self._backend,
                backend_options=dict(self._backend_options),
                config=self._config,
                defaults=_defaults(self._defaults),
                tasks=tasks,
            )
        except CatalogValidationError as exc:
            raise _with_pipeline_name(self.name, exc) from exc
        except ValidationError as exc:
            raise _conf_error(self.name, exc) from exc
        except ValueError as exc:
            raise CatalogValidationError(f"Pipeline '{self.name}': {exc}") from exc

    def _ensure(self) -> KgPipe:
        if self._pipe is not None:
            return self._pipe
        conf = self.conf
        try:
            live = self._execution_tasks(conf)
            if live is None:
                resolved, _steps = resolve_pipeline_tasks(self.name, conf)
            else:
                resolved, _steps = resolve_pipeline_tasks(self.name, conf, tasks=live)
            locations = self._resolve_locations(conf)
            source = _as_data("source", locations["source"], conf, resolved)
            seed = _as_data("seed", locations["seed"], conf, resolved)
            result = _as_data("result", locations["result"], conf, resolved)
            pipe = build_from_conf(self.name, conf, seed, str(locations["data_dir"]), tasks=resolved)
            if locations["ontology"] is not None:
                pipe.ontology_path = str(locations["ontology"])
            pipe.build(source=source, result=result, stable_files=self.stable_files)
        except CatalogValidationError:
            raise
        except ValueError as exc:
            raise CatalogValidationError(f"Pipeline '{self.name}': {exc}") from exc
        self._pipe = pipe
        return pipe

    def _execution_tasks(self, conf: PipelineConf) -> Optional[list[KgTask]]:
        if self._live is not None:
            return list(self._live)
        if not any(isinstance(item, KgTask) for item in self._task_items):
            return None
        try:
            steps = normalize_steps(conf)
        except ValueError as exc:
            raise CatalogValidationError(f"Pipeline '{self.name}': {exc}") from exc
        if len(steps) != len(self._task_items):
            raise CatalogValidationError(
                f"Pipeline '{self.name}' has {len(steps)} steps and {len(self._task_items)} tasks."
            )
        live: list[KgTask] = []
        for item, step in zip(self._task_items, steps):
            if isinstance(item, KgTask):
                if item.name != step.task_name:
                    raise CatalogValidationError(
                        f"Task '{item.name}' does not match spec step '{step.task_name}'."
                    )
                live.append(item)
                continue
            try:
                live.append(Registry.get_task(step.task_name))
            except KeyError as exc:
                raise CatalogValidationError(
                    f"Unknown task '{step.task_name}' in pipeline '{self.name}'. "
                    "Import the task package or run `kgpipe discover` first."
                ) from exc
        return live

    def _resolve_locations(self, conf: PipelineConf) -> dict[str, Any]:
        base = self._base_dir or Path.cwd()
        defaults = conf.defaults.resolve(base_dir=base) if conf.defaults is not None else PipelineDefaults()
        if self._seed is not None and self._target is not None and not _same_file(self._seed, self._target):
            raise CatalogValidationError(
                f"Pipeline '{self.name}': seed and target point at different files. "
                "They are the same input; pass one of them."
            )
        if isinstance(self._data_dir, Data):
            raise CatalogValidationError(f"Pipeline '{self.name}': data_dir must be a directory path.")
        seed_arg = self._seed if self._seed is not None else self._target
        data_dir = _pick_path(self._data_dir, defaults.data_dir, base / ".kgpipe" / self.name, base)
        return {
            "data_dir": data_dir,
            "source": _pick_file(self._source, defaults.source, data_dir / "source", base),
            "seed": _pick_file(seed_arg, defaults.seed, data_dir / "seed", base),
            "result": _pick_file(self._result, defaults.result, data_dir / "result.nt", base),
            "ontology": _pick_ontology(self._ontology, defaults.ontology, base),
        }


def _as_task_list(name: str, tasks: Union[TaskRef, Sequence[TaskRef]]) -> list[TaskRef]:
    if isinstance(tasks, (str, PipelineTaskConf, KgTask)):
        items: list[TaskRef] = [tasks]
    elif isinstance(tasks, Mapping):
        items = [tasks]
    else:
        items = list(tasks)
    if not items:
        raise CatalogValidationError(f"Pipeline '{name}' has no tasks. Add at least one task name or step.")
    return items


def _port(name: str, value: Any) -> PipelinePortDecl:
    if isinstance(value, PipelinePortDecl):
        return value
    if isinstance(value, str):
        return PipelinePortDecl(role=name, format=value)  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        data = dict(value)
        data.setdefault("role", name)
        try:
            return PipelinePortDecl.model_validate(data)
        except ValidationError as exc:
            raise CatalogValidationError(f"Port '{name}': {_validation_detail(exc)}") from exc
    raise CatalogValidationError(f"Port '{name}' must be a format string or a role/format mapping.")


def _defaults(value: Optional[Union[PipelineDefaults, Mapping[str, Any]]]) -> Optional[PipelineDefaults]:
    if value is None:
        return None
    if isinstance(value, PipelineDefaults):
        return value
    if isinstance(value, Mapping):
        try:
            return PipelineDefaults.model_validate(dict(value))
        except ValidationError as exc:
            raise CatalogValidationError(f"`defaults` {_validation_detail(exc)}") from exc
    raise CatalogValidationError("`defaults` must be a mapping of path keys.")


def _conf_error(name: str, exc: ValidationError) -> CatalogValidationError:
    return CatalogValidationError(f"Pipeline '{name}': {_validation_detail(exc)}")


def _with_pipeline_name(name: str, exc: CatalogValidationError) -> CatalogValidationError:
    message = str(exc)
    prefix = f"Pipeline '{name}'"
    if message.startswith(prefix):
        return exc
    return CatalogValidationError(f"{prefix}: {message}")


def _validation_detail(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()))
        message = err.get("msg", "invalid value")
        if err.get("type") == "extra_forbidden":
            field = loc or "unknown"
            parts.append(f"unexpected field '{field}'")
            continue
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts) or "invalid configuration"


def _same_file(left: FileRef, right: FileRef) -> bool:
    left_path = left.path if isinstance(left, Data) else Path(left)
    right_path = right.path if isinstance(right, Data) else Path(right)
    return left_path == right_path


def _pick_path(
    explicit: Optional[Union[str, Path]],
    default: Optional[str],
    fallback: Path,
    base: Path,
) -> Path:
    if explicit is not None:
        return _as_path(explicit, base_dir=base)
    if default:
        return Path(default)
    return fallback


def _pick_file(explicit: Optional[FileRef], default: Optional[str], fallback: Path, base: Path) -> FileRef:
    if isinstance(explicit, Data):
        return explicit
    if explicit is not None:
        return _as_path(explicit, base_dir=base)
    if default:
        return Path(default)
    return fallback


def _pick_ontology(
    explicit: Optional[FileRef],
    default: Optional[str],
    base: Path,
) -> Optional[Path]:
    if isinstance(explicit, Data):
        return explicit.path
    if explicit is not None:
        return _as_path(explicit, base_dir=base)
    if default:
        return Path(default)
    return None


def _as_data(role: str, chosen: FileRef, conf: PipelineConf, tasks: Sequence[Any]) -> Data:
    if isinstance(chosen, Data):
        return chosen
    path = Path(chosen)
    return Data(path=path, format=_format_for_role(conf, role, tasks, path))


def _conf_from_kgpipe(pipe: KgPipe) -> PipelineConf:
    if pipe.step_specs:
        tasks: list[Union[str, PipelineTaskConf]] = [step.model_copy(deep=True) for step in pipe.step_specs]
    else:
        tasks = [task.name for task in pipe.tasks]
    connect: Literal["auto", "explicit"] = "explicit" if pipe.connect == "explicit" else "auto"
    return PipelineConf(
        description=pipe.name or "pipeline",
        connect=connect,
        inputs=dict(pipe.input_decls),
        output=dict(pipe.output_decls),
        tasks=tasks,
    )


def _coerce_build_data(value: FileRef, pipe: KgPipe, *, role: str) -> Data:
    if isinstance(value, Data):
        return value
    path = _as_path(value, base_dir=Path.cwd())
    fmt: Any = None
    if role == "source":
        for decl in pipe.input_decls.values():
            if decl.role == "source":
                fmt = DataFormat(decl.format)
                break
        if fmt is None and pipe.tasks:
            for port, spec in pipe.tasks[0].input_spec.items():
                if port != "kg":
                    fmt = spec
                    break
    else:
        for decl in pipe.output_decls.values():
            fmt = DataFormat(decl.format)
            break
        if fmt is None and pipe.tasks:
            outputs = list(pipe.tasks[-1].output_spec.values())
            if outputs:
                fmt = outputs[0]
    if fmt is None:
        suffix = path.suffix.lower().lstrip(".")
        try:
            fmt = DataFormat.from_extension(suffix) if suffix else DataFormat.RDF_NTRIPLES
        except ValueError:
            fmt = DataFormat.RDF_NTRIPLES
    return Data(path=path, format=fmt)


def _run_status(reports: Sequence[KgTaskReport]) -> Literal["success", "failed", "skipped"]:
    if not reports or any(report.status == "failed" for report in reports):
        return "failed"
    if all(report.status == "skipped" for report in reports):
        return "skipped"
    return "success"
