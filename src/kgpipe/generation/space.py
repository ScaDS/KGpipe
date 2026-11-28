"""Configuration spaces: layouts, task slots, and parameter domains.

A space reuses the categories and parameter constraints declared on the tasks.
The searchable domain is ``Parameter.allowed_values``, or a smaller list that
still has to satisfy those constraints. Sampling and enumeration live here so
a search does not rebuild that logic.

Published parameter-search results stay ``{task_keys, profiles}`` snapshots.
Their file name is :func:`legacy_config_hash`. That digest is independent of
``plan.fingerprint``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from kgpipe.common.model.configuration import (
    ConfigurationProfile,
    Parameter,
    ParameterBinding,
    ParameterType,
)


def category_names(raw: Any) -> List[str]:
    """Normalize a task or search-space category value to a list of names."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        names: List[str] = []
        for item in raw:
            if isinstance(item, str):
                names.append(item)
            else:
                name = getattr(item, "name", None)
                if isinstance(name, str):
                    names.append(name)
        return names
    name = getattr(raw, "name", None)
    if isinstance(name, str):
        return [name]
    return []


def declared_categories(task: Any) -> List[str]:
    return category_names(getattr(task, "category", None))


def lookup_parameter(definition: Any, param_name: str) -> Parameter:
    params = getattr(definition, "parameters", None)
    if params is None:
        raise KeyError(f"Task config_spec has no parameters field (missing {param_name})")
    if hasattr(params, "get"):
        found = params.get(param_name)
        if found is None:
            raise KeyError(f"Parameter {param_name} not found in config_spec.parameters")
        return found
    for param in params:
        if getattr(param, "name", None) == param_name:
            return param
    raise KeyError(f"Parameter {param_name} not found in config_spec.parameters")


def layout_categories(layout: Any) -> List[str]:
    if isinstance(layout, (list, tuple)):
        return [str(item) for item in layout]
    categories = getattr(layout, "allowed_task_categories", None)
    if categories is None:
        raise TypeError("A pipeline layout needs allowed_task_categories.")
    return [str(item) for item in categories]


def legacy_config_hash(snapshot: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical ``{task_keys, profiles}`` snapshot.

    This is the file name of a published result. ``json.dumps(snapshot,
    sort_keys=True)`` is the exact preimage, including the default separators.
    """
    canonical = json.dumps(snapshot, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def legacy_snapshot_key(snapshot: Mapping[str, Any]) -> str:
    """Canonical snapshot text used as the in-memory search identity."""
    return json.dumps(snapshot, sort_keys=True)


@dataclass
class SearchTask:
    """One task that may fill slots in a configuration space."""

    key: str
    task: Any
    categories: List[str]
    domain: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class Candidate:
    """A chosen task order and the parameter profiles bound onto those tasks."""

    task_keys: List[str]
    tasks: List[Any]
    profiles: Dict[str, ConfigurationProfile]


@dataclass
class ConfigurationSpace:
    """Layout plus the tasks allowed to fill it.

    Categories default to the categories declared on each task. A legacy search
    dict can name different slot categories; those are the slots the task
    covers in this space. Parameter domains default to ``allowed_values``.
    An explicit domain is checked against the declared type, allowed values,
    and minimum/maximum.
    """

    tasks: List[SearchTask]
    layout: List[str]
    by_key: Dict[str, SearchTask] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_key = {}
        for entry in self.tasks:
            if entry.key in self.by_key:
                raise ValueError(f"Duplicate task key {entry.key!r}.")
            _validate_domain(entry)
            self.by_key[entry.key] = entry

    @classmethod
    def from_tasks(
        cls,
        tasks: Sequence[Any],
        layout: Any,
        *,
        domains: Optional[Mapping[str, Mapping[str, Sequence[Any]]]] = None,
        categories: Optional[Mapping[str, Sequence[str]]] = None,
        keys: Optional[Mapping[str, str]] = None,
    ) -> "ConfigurationSpace":
        """Build a space from live tasks.

        ``domains`` and ``categories`` are keyed by ``task.name``. ``keys``
        renames a task to the id stored in legacy ``task_keys``.
        """
        domains = domains or {}
        categories = categories or {}
        keys = keys or {}
        entries: List[SearchTask] = []
        for task in tasks:
            name = task.name
            if name in categories:
                cats = [str(item) for item in categories[name]]
            else:
                cats = declared_categories(task)
            if name in domains:
                domain = {param: list(values) for param, values in domains[name].items()}
            else:
                domain = _domain_from_spec(task)
            entries.append(SearchTask(keys.get(name, name), task, cats, domain))
        return cls(entries, layout_categories(layout))

    @classmethod
    def from_legacy(
        cls,
        search_space: Mapping[str, Mapping[str, Any]],
        layout: Any,
        tasks_by_key: Mapping[str, Any],
    ) -> "ConfigurationSpace":
        """Build a space from a ``{task_key: {category, param: [values]}}`` dict.

        Category and domain lists in that dict are data. They are checked
        against the live task; the sampling rules are not copied from it.
        """
        entries: List[SearchTask] = []
        for key, raw in search_space.items():
            if key not in tasks_by_key:
                raise KeyError(key)
            domain: Dict[str, List[Any]] = {}
            for name, values in raw.items():
                if name == "category":
                    continue
                if not isinstance(values, list):
                    raise TypeError(
                        f"Search space values must be lists; got {key}.{name}={type(values)}"
                    )
                domain[name] = list(values)
            entries.append(SearchTask(
                key,
                tasks_by_key[key],
                category_names(raw.get("category")),
                domain,
            ))
        return cls(entries, layout_categories(layout))


def _domain_from_spec(task: Any) -> Dict[str, List[Any]]:
    spec = getattr(task, "config_spec", None)
    if spec is None:
        return {}
    domain: Dict[str, List[Any]] = {}
    for param in spec.parameters:
        if param.allowed_values:
            domain[param.name] = list(param.allowed_values)
    return domain


def _check_domain_value(entry: SearchTask, param: Parameter, value: Any) -> None:
    datatype = param.datatype
    if datatype == ParameterType.number and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise TypeError(
            f"Search domain {entry.key}.{param.name} expects a number, got {value!r}."
        )
    if datatype == ParameterType.integer and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError(
            f"Search domain {entry.key}.{param.name} expects an integer, got {value!r}."
        )
    if datatype == ParameterType.boolean and not isinstance(value, bool):
        raise TypeError(
            f"Search domain {entry.key}.{param.name} expects a boolean, got {value!r}."
        )
    if datatype == ParameterType.string and not isinstance(value, str):
        raise TypeError(
            f"Search domain {entry.key}.{param.name} expects a string, got {value!r}."
        )
    allowed = list(param.allowed_values or [])
    if allowed and value not in allowed:
        raise ValueError(
            f"Search domain {entry.key}.{param.name} value {value!r} "
            f"is outside the allowed values {allowed!r}."
        )
    if param.minimum is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < param.minimum:
            raise ValueError(
                f"Search domain {entry.key}.{param.name} value {value} "
                f"is below the minimum {param.minimum}."
            )
    if param.maximum is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > param.maximum:
            raise ValueError(
                f"Search domain {entry.key}.{param.name} value {value} "
                f"is above the maximum {param.maximum}."
            )


def _validate_domain(entry: SearchTask) -> None:
    spec = getattr(entry.task, "config_spec", None)
    if spec is None:
        if entry.domain:
            raise ValueError(
                f"Task {entry.key!r} has a search domain but does not declare config_spec."
            )
        return
    for name, values in entry.domain.items():
        if not values:
            raise ValueError(f"Empty search space for {entry.key}.{name}")
        try:
            param = lookup_parameter(spec, name)
        except KeyError as exc:
            raise ValueError(f"Parameter {name!r} is not declared on task {entry.key!r}.") from exc
        for value in values:
            _check_domain_value(entry, param, value)


def enumerate_task_combinations(space: ConfigurationSpace) -> List[List[str]]:
    """Task-key orders that fill ``space.layout`` without repeating a category.

    A task is eligible for the current category when that category is one of
    its slots and none of its slots are already covered. A task that covers
    several slots satisfies the later ones, so they are not filled again.
    """
    combos: List[List[str]] = [[]]
    covered_sets: List[set[str]] = [set()]

    for category in space.layout:
        next_combos: List[List[str]] = []
        next_covered: List[set[str]] = []
        for combo, covered in zip(combos, covered_sets):
            if category in covered:
                next_combos.append(combo)
                next_covered.append(covered)
                continue
            eligible: List[SearchTask] = []
            for entry in space.tasks:
                if category not in entry.categories:
                    continue
                if set(entry.categories) & covered:
                    continue
                eligible.append(entry)
            for entry in eligible:
                next_combos.append(combo + [entry.key])
                next_covered.append(set(covered) | set(entry.categories))
        combos, covered_sets = next_combos, next_covered

    seen: set[tuple[str, ...]] = set()
    unique: List[List[str]] = []
    for combo in combos:
        identity = tuple(combo)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(combo)
    return unique


def reject_task_combination(space: ConfigurationSpace, task_keys: Sequence[str]) -> Optional[str]:
    """Return why ``task_keys`` is not a valid layout assignment, or ``None``."""
    keys = list(task_keys)
    unknown = [key for key in keys if key not in space.by_key]
    if unknown:
        return f"Unknown task {unknown[0]!r}."
    if len(keys) != len(set(keys)):
        return f"Duplicate task in {keys!r}."

    covered: set[str] = set()
    for key in keys:
        categories = set(space.by_key[key].categories)
        overlap = categories & covered
        if overlap:
            return (
                f"Task {key!r} repeats categories {sorted(overlap)} "
                f"already covered by the tasks before it."
            )
        covered.update(categories)

    missing = [category for category in space.layout if category not in covered]
    if missing:
        return f"Pipeline does not cover required categories: {missing}."

    valid = {tuple(combo) for combo in enumerate_task_combinations(space)}
    if tuple(keys) not in valid:
        return f"Task keys {keys!r} are not a valid assignment for the layout."
    return None


def _choice(rng: Optional[random.Random]):
    if rng is None:
        return random.choice
    return rng.choice


def _profile_from_bindings(entry: SearchTask, names: Sequence[str], values: Sequence[Any]) -> Optional[ConfigurationProfile]:
    spec = getattr(entry.task, "config_spec", None)
    if spec is None or not names:
        return None
    bindings = [
        ParameterBinding(parameter=lookup_parameter(spec, name), value=value)
        for name, value in zip(names, values)
    ]
    label = ",".join(f"{name}={value}" for name, value in zip(names, values))
    return ConfigurationProfile(
        name=f"{entry.task.name}_{label}",
        definition=spec,
        bindings=bindings,
    )


def sample_profile(entry: SearchTask, rng: Optional[random.Random] = None) -> Optional[ConfigurationProfile]:
    """Draw one value per domain parameter, in domain order."""
    draw = _choice(rng)
    names: List[str] = []
    values: List[Any] = []
    for name, options in entry.domain.items():
        names.append(name)
        values.append(draw(options))
    return _profile_from_bindings(entry, names, values)


def sample_candidate(space: ConfigurationSpace, rng: Optional[random.Random] = None) -> Candidate:
    """Draw one valid candidate. The same ``rng`` seed draws the same candidate."""
    draw = _choice(rng)
    keys: List[str] = []
    tasks: List[Any] = []
    profiles: Dict[str, ConfigurationProfile] = {}
    covered: set[str] = set()

    for category in space.layout:
        if category in covered:
            continue
        eligible = [entry for entry in space.tasks if category in entry.categories]
        if not eligible:
            continue
        eligible = [entry for entry in eligible if not (set(entry.categories) & covered)]
        if not eligible:
            raise ValueError(
                f"No task can cover category {category!r} without overlapping already covered "
                f"categories {sorted(covered)}. Adjust search_space or pipeline_layout."
            )
        entry = draw(eligible)
        covered.update(entry.categories)
        keys.append(entry.key)
        tasks.append(entry.task)
        profile = sample_profile(entry, rng)
        if profile is not None:
            profiles[entry.task.name] = profile

    return Candidate(keys, tasks, profiles)


def sample_for_combo(
    space: ConfigurationSpace,
    task_keys: Sequence[str],
    rng: random.Random,
    *,
    template_keys: Optional[Sequence[str]] = None,
    template_profiles: Optional[Mapping[str, ConfigurationProfile]] = None,
) -> Candidate:
    """Sample parameter values for a fixed task order.

    A template profile is reused when the task key at that index is unchanged.
    Reused profiles do not consume ``rng``.
    """
    template_keys = list(template_keys or [])
    template_profiles = template_profiles or {}
    keys = list(task_keys)
    tasks: List[Any] = []
    profiles: Dict[str, ConfigurationProfile] = {}
    for index, key in enumerate(keys):
        entry = space.by_key[key]
        tasks.append(entry.task)
        if index < len(template_keys) and template_keys[index] == key:
            existing = template_profiles.get(entry.task.name)
            if existing is not None:
                profiles[entry.task.name] = existing
                continue
        profile = sample_profile(entry, rng)
        if profile is not None:
            profiles[entry.task.name] = profile
    return Candidate(keys, tasks, profiles)


def _assignments(entry: SearchTask) -> List[Dict[str, Any]]:
    if not entry.domain:
        return [{}]
    keys = list(entry.domain)
    lists = [entry.domain[key] for key in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*lists)]


def candidate_for_assignment(
    space: ConfigurationSpace,
    task_keys: Sequence[str],
    assignment: Sequence[Mapping[str, Any]],
) -> Candidate:
    keys = list(task_keys)
    tasks: List[Any] = []
    profiles: Dict[str, ConfigurationProfile] = {}
    for key, params in zip(keys, assignment):
        entry = space.by_key[key]
        tasks.append(entry.task)
        names = [name for name in entry.domain if name in params]
        values = [params[name] for name in names]
        profile = _profile_from_bindings(entry, names, values)
        if profile is not None:
            profiles[entry.task.name] = profile
    return Candidate(keys, tasks, profiles)


def snapshots_for_combo(space: ConfigurationSpace, task_keys: Sequence[str]) -> List[Dict[str, Any]]:
    per_task = [_assignments(space.by_key[key]) for key in task_keys]
    snapshots: List[Dict[str, Any]] = []
    for assignment in itertools.product(*per_task):
        candidate = candidate_for_assignment(space, task_keys, assignment)
        snapshots.append(candidate_snapshot(candidate))
    return snapshots


def enumerate_snapshots(space: ConfigurationSpace) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    for combo in enumerate_task_combinations(space):
        snapshots.extend(snapshots_for_combo(space, combo))
    return snapshots


def enumerate_candidates(space: ConfigurationSpace) -> List[Candidate]:
    candidates: List[Candidate] = []
    for combo in enumerate_task_combinations(space):
        per_task = [_assignments(space.by_key[key]) for key in combo]
        for assignment in itertools.product(*per_task):
            candidates.append(candidate_for_assignment(space, combo, assignment))
    return candidates


def sample_unique_snapshots(
    space: ConfigurationSpace,
    *,
    n: int,
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample up to ``n`` distinct snapshots for each valid task combination."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    combos = enumerate_task_combinations(space)
    snapshots: List[Dict[str, Any]] = []
    combo_stats: List[Dict[str, Any]] = []
    for combo in combos:
        available = snapshots_for_combo(space, combo)
        serialized = [legacy_snapshot_key(item) for item in available]
        if len(set(serialized)) != len(serialized):
            raise ValueError(f"Duplicate profile snapshots for combo {combo!r}")
        sample_count = min(n, len(available))
        picked = rng.sample(available, k=sample_count) if sample_count else []
        snapshots.extend(picked)
        combo_stats.append({
            "task_keys": combo,
            "available_profiles": len(available),
            "requested": n,
            "sampled": sample_count,
            "exhausted": sample_count < n,
        })
    stats = {
        "requested_n": n,
        "total_combos": len(combos),
        "total_snapshots": len(snapshots),
        "combos_exhausted": sum(1 for row in combo_stats if row["exhausted"]),
        "combos": combo_stats,
    }
    return snapshots, stats


def candidate_snapshot(candidate: Candidate) -> Dict[str, Any]:
    """Legacy ``{task_keys, profiles}`` snapshot. The hash of this object is the result key."""
    profiles: Dict[str, Any] = {}
    for task in candidate.tasks:
        profile = candidate.profiles.get(task.name)
        if profile is None:
            continue
        profiles[task.name] = {
            "profile_name": profile.name,
            "bindings": [
                {"parameter": binding.parameter.name, "value": binding.value}
                for binding in profile.bindings
            ],
        }
    return {"task_keys": list(candidate.task_keys), "profiles": profiles}


def candidate_from_snapshot(snapshot: Mapping[str, Any], tasks_by_key: Mapping[str, Any]) -> Candidate:
    """Rebuild a candidate from a legacy snapshot. Profile names and binding order are kept."""
    task_keys = list(snapshot["task_keys"])
    raw_profiles = snapshot.get("profiles") or {}
    tasks: List[Any] = []
    profiles: Dict[str, ConfigurationProfile] = {}
    for key in task_keys:
        task = tasks_by_key[key]
        tasks.append(task)
        raw = raw_profiles.get(task.name)
        spec = getattr(task, "config_spec", None)
        if raw is None or spec is None:
            continue
        bindings = [
            ParameterBinding(
                parameter=lookup_parameter(spec, item["parameter"]),
                value=item["value"],
            )
            for item in raw["bindings"]
        ]
        profiles[task.name] = ConfigurationProfile(
            name=raw["profile_name"],
            definition=spec,
            bindings=bindings,
        )
    return Candidate(task_keys, tasks, profiles)


def load_legacy_sample_file(path: Path, *, expected_version: int = 1, label: str = "pipeline config") -> List[Dict[str, Any]]:
    """Read a versioned fixture ``{version, samples: [snapshot, ...]}``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != expected_version:
        raise ValueError(
            f"Unsupported {label} snapshot version {raw.get('version')!r}; "
            f"expected {expected_version}"
        )
    samples = raw.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path} has no samples list.")
    return samples


def read_legacy_snapshot(path: Path, *, check_hash: bool = True) -> Dict[str, Any]:
    """Read one published ``{task_keys, profiles}`` file.

    With ``check_hash``, the file stem must be :func:`legacy_config_hash`.
    """
    path = Path(path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or "task_keys" not in snapshot:
        raise ValueError(f"{path} is not a legacy configuration snapshot.")
    if check_hash and path.stem != legacy_config_hash(snapshot):
        raise ValueError(
            f"{path.name} does not match the legacy config hash {legacy_config_hash(snapshot)}."
        )
    return snapshot


def iter_legacy_result_snapshots(directory: Path) -> Iterator[tuple[Path, Dict[str, Any]]]:
    """Yield ``(path, snapshot)`` for ``{hash}.json`` files in a result directory.

    ``*.plan.json``, ``*.eval.json``, and ``results.json`` are not configuration snapshots.
    """
    directory = Path(directory)
    for path in sorted(directory.glob("*.json")):
        if path.suffixes != [".json"]:
            continue
        stem = path.stem
        if len(stem) != 64 or any(char not in "0123456789abcdef" for char in stem):
            continue
        yield path, read_legacy_snapshot(path)
