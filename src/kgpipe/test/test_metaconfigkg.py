"""
In-memory (collections/dicts) CRUD backend for the Meta-Ontology, with full Pydantic models.

- Pydantic v2 is assumed. A small compatibility shim is included for v1-style .dict().
- Storage: per-entity dict {id -> model_dump()}
- CRUD: create/get/update/delete/list (+ exists, upsert, bulk)
- Query: simple equality filters + sort + cursor pagination

You can extend:
- richer filtering (gt/lt/contains), graph traversals, secondary indexes
- validation constraints, unique keys, referential integrity checks
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


# ----------------------------
# Pydantic helpers / types
# ----------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def model_to_dict(m: BaseModel) -> Dict[str, Any]:
    # Pydantic v2
    if hasattr(m, "model_dump"):
        return m.model_dump()
    # Pydantic v1 fallback
    return m.dict()  # type: ignore[attr-defined]


def model_copy_update(m: BaseModel, patch: Mapping[str, Any]) -> BaseModel:
    # Pydantic v2
    if hasattr(m, "model_copy"):
        return m.model_copy(update=dict(patch))
    # Pydantic v1 fallback
    return m.copy(update=dict(patch))  # type: ignore[attr-defined]


class SortDir(str, Enum):
    asc = "asc"
    desc = "desc"


# ----------------------------
# Core enums
# ----------------------------

class ArtifactKind(str, Enum):
    python_package = "python_package"
    java_application = "java_application"
    library = "library"
    docker_image = "docker_image"
    rest_service = "rest_service"
    spark_job = "spark_job"
    other = "other"


class ComponentKind(str, Enum):
    cli_command = "cli_command"
    rest_endpoint = "rest_endpoint"
    java_class = "java_class"
    python_callable = "python_callable"
    spark_job = "spark_job"
    other = "other"


class ParameterDataType(str, Enum):
    boolean = "boolean"
    integer = "integer"
    number = "number"      # float
    string = "string"
    enum = "enum"
    array = "array"
    object = "object"


class ParameterScope(str, Enum):
    training = "training"
    inference = "inference"
    io = "io"
    resources = "resources"
    general = "general"


class BindingStrength(str, Enum):
    exact = "exact"
    approximate = "approximate"
    heuristic = "heuristic"


class MetricKind(str, Enum):
    quality = "quality"
    performance = "performance"
    resource = "resource"
    other = "other"


# ----------------------------
# Base entity model
# ----------------------------

class EntityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def touch(self) -> "EntityModel":
        return model_copy_update(self, {"updated_at": utcnow()})  # type: ignore[return-value]


# ----------------------------
# Task layer
# ----------------------------

class TaskCategory(EntityModel):
    id: str = Field(default_factory=lambda: new_id("taskcat"))
    name: str
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)


class Task(EntityModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    name: str
    description: Optional[str] = None
    task_category_id: str
    # Optional pipeline semantics:
    precedes_task_ids: List[str] = Field(default_factory=list)


# ----------------------------
# Implementation layer
# ----------------------------

class Tool(EntityModel):
    id: str = Field(default_factory=lambda: new_id("tool"))
    name: str
    kind: ArtifactKind = ArtifactKind.other
    homepage: Optional[str] = None
    repository: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # What it claims to support (conceptual mapping)
    implements_task_category_ids: List[str] = Field(default_factory=list)


class ToolRelease(EntityModel):
    id: str = Field(default_factory=lambda: new_id("release"))
    tool_id: str

    version: str  # e.g. "1.4.2"
    release_date: Optional[datetime] = None

    # execution environment hints
    language: Optional[str] = None            # "python", "java", "scala"
    runtime: Optional[str] = None             # "python>=3.10", "java17"
    license: Optional[str] = None

    # e.g. docker image ref, pypi name, maven coords
    artifact_ref: Optional[str] = None


class Component(EntityModel):
    id: str = Field(default_factory=lambda: new_id("component"))
    release_id: str

    name: str
    kind: ComponentKind = ComponentKind.other
    description: Optional[str] = None

    # Conceptual: which task does this runnable unit implement?
    implements_task_id: str

    # How to invoke (one of these typically)
    cli_command: Optional[str] = None                 # e.g. "dedupe train"
    rest_method: Optional[Literal["GET", "POST", "PUT", "DELETE", "PATCH"]] = None
    rest_path: Optional[str] = None                   # e.g. "/predict"
    entrypoint: Optional[str] = None                  # e.g. "package.module:func"


# ----------------------------
# Configuration layer
# ----------------------------

class Parameter(EntityModel):
    id: str = Field(default_factory=lambda: new_id("param"))
    component_id: str

    # raw/native name(s)
    name: str                         # canonical within the tool
    native_keys: List[str] = Field(default_factory=list)  # e.g. ["--threshold", "threshold"]

    description: Optional[str] = None
    scope: ParameterScope = ParameterScope.general

    datatype: ParameterDataType
    required: bool = False
    default_value: Optional[Any] = None

    # Enum constraints if datatype == enum
    allowed_values: Optional[List[Any]] = None

    # Numeric constraints (if applicable)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None


class ConfigurationProfile(EntityModel):
    id: str = Field(default_factory=lambda: new_id("profile"))
    name: str
    description: Optional[str] = None

    # Which runnable unit this profile targets (strongly recommended)
    component_id: str

    tags: List[str] = Field(default_factory=list)


class ParameterSetting(EntityModel):
    id: str = Field(default_factory=lambda: new_id("pset"))
    profile_id: str
    parameter_id: str
    value: Any

    # provenance
    source: Optional[str] = None  # "docs", "experiment", "user", etc.


# ----------------------------
# Canonical semantics layer
# ----------------------------

class OptionCategory(EntityModel):
    id: str = Field(default_factory=lambda: new_id("optcat"))
    name: str
    description: Optional[str] = None


class CanonicalOption(EntityModel):
    id: str = Field(default_factory=lambda: new_id("copt"))
    name: str
    description: Optional[str] = None

    option_category_id: str
    expected_datatype: ParameterDataType
    expected_unit: Optional[str] = None
    typical_min: Optional[float] = None
    typical_max: Optional[float] = None
    aliases: List[str] = Field(default_factory=list)


class ParameterBinding(EntityModel):
    """
    Reifies alignment between tool-specific Parameter and CanonicalOption.
    Allows exact/approx/heuristic mappings + transforms/units.
    """
    id: str = Field(default_factory=lambda: new_id("bind"))
    parameter_id: str
    canonical_option_id: str

    strength: BindingStrength = BindingStrength.exact

    # transforms are intentionally strings: you can store a rule, a JSONata, a python expr, etc.
    value_transform: Optional[str] = None  # e.g. "x -> 1 - x"
    unit_transform: Optional[str] = None   # e.g. "ms -> s : x/1000"

    constraints_note: Optional[str] = None
    documentation_source: Optional[str] = None
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ----------------------------
# Execution / provenance layer
# ----------------------------

class Dataset(EntityModel):
    id: str = Field(default_factory=lambda: new_id("ds"))
    name: str
    description: Optional[str] = None
    uri: Optional[str] = None


class DataProfile(EntityModel):
    id: str = Field(default_factory=lambda: new_id("dprof"))
    dataset_id: str

    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    missingness_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = None


class Metric(EntityModel):
    id: str = Field(default_factory=lambda: new_id("metric"))
    name: str                         # e.g. "f1", "precision", "runtime_seconds"
    kind: MetricKind = MetricKind.other
    unit: Optional[str] = None


class ExecutionRun(EntityModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    component_id: str
    profile_id: Optional[str] = None  # might run ad-hoc without a stored profile
    dataset_id: Optional[str] = None

    started_at: datetime = Field(default_factory=utcnow)
    ended_at: Optional[datetime] = None
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"

    # record environment info
    tool_release_id: Optional[str] = None
    host: Optional[str] = None
    notes: Optional[str] = None

    # observed results (simple map; you can also normalize as MetricObservation entities)
    metric_values: Dict[str, Any] = Field(default_factory=dict)  # metric_id -> value


# ----------------------------
# Query & pagination
# ----------------------------

@dataclass(frozen=True)
class Query:
    filters: Mapping[str, Any] = None
    sort: Sequence[Tuple[str, SortDir]] = ()
    limit: int = 50
    cursor: Optional[str] = None  # opaque string; here we use an integer offset


@dataclass(frozen=True)
class Page(Generic[TypeVar("T")]):
    items: List[Any]
    next_cursor: Optional[str] = None
    total: Optional[int] = None


# ----------------------------
# In-memory CRUD repository
# ----------------------------

TModel = TypeVar("TModel", bound=EntityModel)


class NotFoundError(KeyError):
    pass


class ConflictError(ValueError):
    pass


class InMemoryRepository(Generic[TModel]):
    """
    Simple backend:
      - stores model dicts by id
      - returns parsed Pydantic models
      - supports patch updates + list with filtering/sorting/cursor pagination
    """

    def __init__(self, model_cls: Type[TModel], *, name: str):
        self._model_cls = model_cls
        self._name = name
        self._store: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def create(self, obj: Union[TModel, Mapping[str, Any]]) -> TModel:
        model = obj if isinstance(obj, self._model_cls) else self._model_cls(**dict(obj))
        obj_id = model.id
        if obj_id in self._store:
            raise ConflictError(f"{self._name}: id already exists: {obj_id}")
        self._store[obj_id] = model_to_dict(model)
        return self._model_cls(**self._store[obj_id])

    def get(self, obj_id: str) -> Optional[TModel]:
        data = self._store.get(obj_id)
        return self._model_cls(**data) if data else None

    def require(self, obj_id: str) -> TModel:
        obj = self.get(obj_id)
        if obj is None:
            raise NotFoundError(f"{self._name}: not found: {obj_id}")
        return obj

    def update(self, obj_id: str, patch: Mapping[str, Any]) -> TModel:
        current = self.require(obj_id)
        patched = model_copy_update(current, patch)
        # update updated_at unless caller explicitly set it
        if "updated_at" not in patch:
            patched = model_copy_update(patched, {"updated_at": utcnow()})
        self._store[obj_id] = model_to_dict(patched)
        return self._model_cls(**self._store[obj_id])

    def delete(self, obj_id: str) -> bool:
        return self._store.pop(obj_id, None) is not None

    def exists(self, obj_id: str) -> bool:
        return obj_id in self._store

    def upsert(self, obj: Union[TModel, Mapping[str, Any]]) -> TModel:
        model = obj if isinstance(obj, self._model_cls) else self._model_cls(**dict(obj))
        if self.exists(model.id):
            # Replace-ish via patch of all fields
            return self.update(model.id, model_to_dict(model))
        return self.create(model)

    def bulk_create(self, objs: Iterable[Union[TModel, Mapping[str, Any]]]) -> List[TModel]:
        return [self.create(o) for o in objs]

    def bulk_delete(self, obj_ids: Iterable[str]) -> int:
        n = 0
        for oid in obj_ids:
            if self.delete(oid):
                n += 1
        return n

    def list(self, query: Query = Query()) -> Page[TModel]:
        filters = query.filters or {}
        items = [self._model_cls(**v) for v in self._store.values()]

        # filtering: equality + simple "contains" for list fields
        def matches(m: TModel) -> bool:
            for k, expected in filters.items():
                if not hasattr(m, k):
                    return False
                actual = getattr(m, k)

                # list containment
                if isinstance(actual, list) and not isinstance(expected, list):
                    if expected not in actual:
                        return False
                else:
                    if actual != expected:
                        return False
            return True

        items = [m for m in items if matches(m)]

        # sorting
        for field, direction in reversed(list(query.sort)):
            reverse = direction == SortDir.desc
            items.sort(key=lambda x: getattr(x, field, None), reverse=reverse)

        total = len(items)

        # cursor pagination using offset
        offset = int(query.cursor) if query.cursor else 0
        limit = max(0, int(query.limit))
        page_items = items[offset : offset + limit]

        next_offset = offset + limit
        next_cursor = str(next_offset) if next_offset < total else None

        return Page(items=page_items, next_cursor=next_cursor, total=total)


# ----------------------------
# Repository registry (all entity repos)
# ----------------------------

@dataclass
class Repositories:
    task_categories: InMemoryRepository[TaskCategory]
    tasks: InMemoryRepository[Task]
    tools: InMemoryRepository[Tool]
    releases: InMemoryRepository[ToolRelease]
    components: InMemoryRepository[Component]
    parameters: InMemoryRepository[Parameter]
    option_categories: InMemoryRepository[OptionCategory]
    canonical_options: InMemoryRepository[CanonicalOption]
    parameter_bindings: InMemoryRepository[ParameterBinding]
    config_profiles: InMemoryRepository[ConfigurationProfile]
    parameter_settings: InMemoryRepository[ParameterSetting]
    datasets: InMemoryRepository[Dataset]
    data_profiles: InMemoryRepository[DataProfile]
    metrics: InMemoryRepository[Metric]
    runs: InMemoryRepository[ExecutionRun]


def make_repositories() -> Repositories:
    return Repositories(
        task_categories=InMemoryRepository(TaskCategory, name="TaskCategory"),
        tasks=InMemoryRepository(Task, name="Task"),
        tools=InMemoryRepository(Tool, name="Tool"),
        releases=InMemoryRepository(ToolRelease, name="ToolRelease"),
        components=InMemoryRepository(Component, name="Component"),
        parameters=InMemoryRepository(Parameter, name="Parameter"),
        option_categories=InMemoryRepository(OptionCategory, name="OptionCategory"),
        canonical_options=InMemoryRepository(CanonicalOption, name="CanonicalOption"),
        parameter_bindings=InMemoryRepository(ParameterBinding, name="ParameterBinding"),
        config_profiles=InMemoryRepository(ConfigurationProfile, name="ConfigurationProfile"),
        parameter_settings=InMemoryRepository(ParameterSetting, name="ParameterSetting"),
        datasets=InMemoryRepository(Dataset, name="Dataset"),
        data_profiles=InMemoryRepository(DataProfile, name="DataProfile"),
        metrics=InMemoryRepository(Metric, name="Metric"),
        runs=InMemoryRepository(ExecutionRun, name="ExecutionRun"),
    )


# ----------------------------
# (Optional) Domain-specific helper functions
# ----------------------------

def list_bindings_by_parameter(
    repos: Repositories, parameter_id: str, query: Query = Query()
) -> Page[ParameterBinding]:
    q = Query(
        filters={**(query.filters or {}), "parameter_id": parameter_id},
        sort=query.sort,
        limit=query.limit,
        cursor=query.cursor,
    )
    return repos.parameter_bindings.list(q)


def list_bindings_by_canonical_option(
    repos: Repositories, canonical_option_id: str, query: Query = Query()
) -> Page[ParameterBinding]:
    q = Query(
        filters={**(query.filters or {}), "canonical_option_id": canonical_option_id},
        sort=query.sort,
        limit=query.limit,
        cursor=query.cursor,
    )
    return repos.parameter_bindings.list(q)


def list_parameter_settings_by_profile(
    repos: Repositories, profile_id: str, query: Query = Query()
) -> Page[ParameterSetting]:
    q = Query(
        filters={**(query.filters or {}), "profile_id": profile_id},
        sort=query.sort,
        limit=query.limit,
        cursor=query.cursor,
    )
    return repos.parameter_settings.list(q)


# ----------------------------
# Example usage (you can delete this)
# ----------------------------

if __name__ == "__main__":
    repos = make_repositories()

    # Create task category + task
    em = repos.task_categories.create({"name": "EntityMatching", "description": "Record linkage / dedup"})
    blocking = repos.tasks.create({"name": "Blocking", "task_category_id": em.id})

    # Tool + release + component
    tool = repos.tools.create({"name": "MyMatcher", "kind": "python_package", "implements_task_category_ids": [em.id]})
    rel = repos.releases.create({"tool_id": tool.id, "version": "1.0.0", "language": "python", "runtime": "python>=3.10"})
    comp = repos.components.create(
        {"release_id": rel.id, "name": "match-cli", "kind": "cli_command", "implements_task_id": blocking.id, "cli_command": "mymatcher match"}
    )

    # Parameter + canonical option + binding
    optcat = repos.option_categories.create({"name": "Similarity"})
    copt = repos.canonical_options.create(
        {"name": "SimilarityThreshold", "option_category_id": optcat.id, "expected_datatype": "number", "typical_min": 0.0, "typical_max": 1.0}
    )
    param = repos.parameters.create(
        {"component_id": comp.id, "name": "threshold", "native_keys": ["--threshold"], "datatype": "number", "default_value": 0.5, "scope": "inference"}
    )
    bind = repos.parameter_bindings.create(
        {"parameter_id": param.id, "canonical_option_id": copt.id, "strength": "exact", "confidence_score": 0.95}
    )

    # Configuration profile + setting
    profile = repos.config_profiles.create({"name": "high-precision", "component_id": comp.id})
    repos.parameter_settings.create({"profile_id": profile.id, "parameter_id": param.id, "value": 0.85, "source": "experiment"})

    # List common: bindings for canonical option
    page = list_bindings_by_canonical_option(repos, copt.id, Query(limit=10))
    print("Bindings for option:", copt.name, [b.id for b in page.items])
