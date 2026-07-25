from typing import List, Dict, Any, Optional
from kgpipe.common import KgTask
from kgpipe.common.model.configuration import ConfigurationProfile, ParameterBinding
from kgpipe_search.definitions import (
    PipelineLayout,
    PipelineConfig,
    RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE,
    RDF_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE,
    RDF_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE,
    TEXT_SAMPLED_PIPELINE_CONFIGS_FIXTURE,
    TEXT_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE,
    TEXT_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE,
    _RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _RDF_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _RDF_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _TEXT_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _TEXT_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _TEXT_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
)
import json
import random
import itertools
from pathlib import Path
from kgpipe_search.definitions import task_dict


def _task_categories_list(search_space: Dict[str, Dict[str, Any]], task_name: str) -> List[str]:
    raw = search_space.get(task_name, {}).get("category")
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, str)]
    if isinstance(raw, str):
        return [raw]
    return []


def enumerate_valid_task_combinations(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
) -> List[List[str]]:
    """
    Enumerate all possible task-name combinations for the given pipeline layout,
    respecting category order and multi-category coverage, without sampling config options.

    A task is only eligible for the current category if its declared categories are
    disjoint from categories already covered by earlier tasks. That avoids pairing e.g.
    Paris ontology matching with a dual-category embedding matcher that would repeat
    ontology coverage when only entity matching is still needed.
    """
    all_task_names = list(search_space.keys())

    combos: List[List[str]] = [[]]
    covered_sets: List[set[str]] = [set()]

    for category in pipeline_layout.allowed_task_categories:
        next_combos: List[List[str]] = []
        next_covered_sets: List[set[str]] = []

        for combo, covered in zip(combos, covered_sets):
            if category in covered:
                next_combos.append(combo)
                next_covered_sets.append(covered)
                continue

            eligible: List[str] = []
            for tn in all_task_names:
                cats = _task_categories_list(search_space, tn)
                if category not in cats:
                    continue
                if set(cats) & covered:
                    continue
                eligible.append(tn)
            for tn in eligible:
                new_combo = combo + [tn]
                new_covered = set(covered)
                new_covered.update(_task_categories_list(search_space, tn))
                next_combos.append(new_combo)
                next_covered_sets.append(new_covered)

        combos, covered_sets = next_combos, next_covered_sets

    # De-duplicate while keeping stable order.
    seen: set[tuple[str, ...]] = set()
    unique: List[List[str]] = []
    for c in combos:
        t = tuple(c)
        if t in seen:
            continue
        seen.add(t)
        unique.append(c)
    return unique

def _get_param(definition: Any, param_name: str):
    params = getattr(definition, "parameters", None)
    if params is None:
        raise KeyError(f"Task config_spec has no parameters field (missing {param_name})")

    # common shapes: dict-like or list of Parameter
    if hasattr(params, "get"):
        p = params.get(param_name)
        if p is None:
            raise KeyError(f"Parameter {param_name} not found in config_spec.parameters")
        return p

    for p in params:
        if getattr(p, "name", None) == param_name:
            return p
    raise KeyError(f"Parameter {param_name} not found in config_spec.parameters")


def pipeline_config_to_snapshot(task_keys: List[str], pipeline_config: PipelineConfig) -> Dict[str, Any]:
    profiles: Dict[str, Any] = {}
    for task in pipeline_config.tasks:
        prof = pipeline_config.config_catalog.get(task.name)
        if prof is None:
            continue
        profiles[task.name] = {
            "profile_name": prof.name,
            "bindings": [
                {"parameter": binding.parameter.name, "value": binding.value}
                for binding in prof.bindings
            ],
        }
    return {"task_keys": task_keys, "profiles": profiles}


def pipeline_config_from_snapshot(snapshot: Dict[str, Any]) -> PipelineConfig:
    task_keys: List[str] = snapshot["task_keys"]
    profiles: Dict[str, Any] = snapshot.get("profiles") or {}
    tasks: List[KgTask] = []
    config_catalog: Dict[str, ConfigurationProfile] = {}

    for task_key in task_keys:
        task = task_dict[task_key]
        tasks.append(task)
        prof_data = profiles.get(task.name)
        if prof_data is None:
            continue
        if getattr(task, "config_spec", None) is None:
            continue
        bindings = [
            ParameterBinding(
                parameter=_get_param(task.config_spec, b["parameter"]),
                value=b["value"],
            )
            for b in prof_data["bindings"]
        ]
        config_catalog[task.name] = ConfigurationProfile(
            name=prof_data["profile_name"],
            definition=task.config_spec,
            bindings=bindings,
        )

    return PipelineConfig(tasks=tasks, config_catalog=config_catalog)


def load_rdf_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if raw.get("version") != _RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported rdf sampled configs snapshot version {raw.get('version')!r}; "
            f"expected {_RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION}"
        )
    return [pipeline_config_from_snapshot(item) for item in raw["samples"]]


def load_text_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or TEXT_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if raw.get("version") != _TEXT_PIPELINE_CONFIG_SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported text sampled configs snapshot version {raw.get('version')!r}; "
            f"expected {_TEXT_PIPELINE_CONFIG_SNAPSHOT_VERSION}"
        )
    return [pipeline_config_from_snapshot(item) for item in raw["samples"]]


def load_rdf_unique_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or RDF_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if raw.get("version") != _RDF_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported rdf unique sampled configs snapshot version {raw.get('version')!r}; "
            f"expected {_RDF_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION}"
        )
    return [pipeline_config_from_snapshot(item) for item in raw["samples"]]


def load_text_unique_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or TEXT_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if raw.get("version") != _TEXT_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported text unique sampled configs snapshot version {raw.get('version')!r}; "
            f"expected {_TEXT_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION}"
        )
    return [pipeline_config_from_snapshot(item) for item in raw["samples"]]


def load_rdf_exhaustive_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or RDF_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if raw.get("version") != _RDF_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported rdf exhaustive configs snapshot version {raw.get('version')!r}; "
            f"expected {_RDF_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION}"
        )
    return [pipeline_config_from_snapshot(item) for item in raw["samples"]]


def load_text_exhaustive_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or TEXT_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if raw.get("version") != _TEXT_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported text exhaustive configs snapshot version {raw.get('version')!r}; "
            f"expected {_TEXT_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION}"
        )
    return [pipeline_config_from_snapshot(item) for item in raw["samples"]]


# TODO rules for valid pipeline config:
def sample_valid_pipeline_config(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    rng: Optional[random.Random] = None,
) -> PipelineConfig:
    """
    Randomly sample a valid pipeline config from the search space,
    respecting the order of categories in the pipeline layout.
    """
    draw = rng.choice if rng is not None else random.choice
    tasks: List[KgTask] = []
    config_catalog: Dict[str, ConfigurationProfile] = {}
    covered_categories: set[str] = set()

    for category in pipeline_layout.allowed_task_categories:
        if category in covered_categories:
            continue

        eligible_task_names = [
            tn
            for tn, space in search_space.items()
            if (
                space.get("category") == category
                or (
                    isinstance(space.get("category"), list)
                    and category in (space.get("category") or [])
                )
            )
        ]
        if not eligible_task_names:
            continue

        eligible_task_names = [
            tn
            for tn in eligible_task_names
            if not (set(_task_categories_list(search_space, tn)) & covered_categories)
        ]
        if not eligible_task_names:
            raise ValueError(
                f"No task can cover category {category!r} without overlapping already covered "
                f"categories {sorted(covered_categories)}. Adjust search_space or pipeline_layout."
            )

        task_key = draw(eligible_task_names)
        task = task_dict[task_key]
        covered_categories.update(_task_categories_list(search_space, task_key))
        tasks.append(task)

        # metadata only or task has no config spec
        if getattr(task, "config_spec", None) is None:
            continue

        bindings: List[ParameterBinding] = []
        name_parts: List[str] = []
        for config_name, config_values in search_space[task_key].items():
            if config_name == "category":
                continue
            if not isinstance(config_values, list):
                raise TypeError(
                    f"Search space values must be lists; got {task_key}.{config_name}={type(config_values)}"
                )
            if not config_values:
                raise ValueError(f"Empty search space for {task_key}.{config_name}")

            config_value = draw(config_values)
            name_parts.append(f"{config_name}={config_value}")
            bindings.append(
                ParameterBinding(
                    parameter=_get_param(task.config_spec, config_name),
                    value=config_value,
                )
            )

        if bindings:
            config_catalog[task.name] = ConfigurationProfile(
                name=f"{task.name}_" + ",".join(name_parts),
                definition=task.config_spec,
                bindings=bindings,
            )

    return PipelineConfig(tasks=tasks, config_catalog=config_catalog)


_PIPELINE_CONFIG_SNAPSHOT_FILE_VERSION = 1


def save_pipeline_config_snapshot(
    path: Path,
    pipeline_config: PipelineConfig,
    *,
    task_keys: Optional[List[str]] = None,
) -> None:
    keys = task_keys or task_keys_from_pipeline_config(pipeline_config)
    payload = {
        "version": _PIPELINE_CONFIG_SNAPSHOT_FILE_VERSION,
        "snapshot": pipeline_config_to_snapshot(keys, pipeline_config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_pipeline_config_snapshot(path: Path) -> PipelineConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != _PIPELINE_CONFIG_SNAPSHOT_FILE_VERSION:
        raise ValueError(
            f"Unsupported pipeline config snapshot file version {raw.get('version')!r}; "
            f"expected {_PIPELINE_CONFIG_SNAPSHOT_FILE_VERSION}"
        )
    return pipeline_config_from_snapshot(raw["snapshot"])


def task_keys_from_pipeline_config(pipeline_config: PipelineConfig) -> List[str]:
    keys: List[str] = []
    for task in pipeline_config.tasks:
        for task_key, registered in task_dict.items():
            if registered is task or registered.name == task.name:
                keys.append(task_key)
                break
        else:
            raise ValueError(f"Unknown task {task.name!r}")
    return keys


def pipeline_config_snapshot_key(
    pipeline_config: PipelineConfig,
    search_space: Dict[str, Dict[str, Any]],
) -> str:
    task_keys = task_keys_from_pipeline_config(pipeline_config)
    snapshot = pipeline_config_to_snapshot(task_keys, pipeline_config)
    return json.dumps(snapshot, sort_keys=True)


def build_pipeline_config_for_task_combo(
    search_space: Dict[str, Dict[str, Any]],
    task_name_combo: List[str],
    *,
    rng: random.Random,
    template: Optional[PipelineConfig] = None,
) -> PipelineConfig:
    """
    Build a pipeline config for a fixed task combo.
    Reuses parameter profiles from template when the task key is unchanged.
    """
    template_keys = (
        task_keys_from_pipeline_config(template) if template is not None else []
    )
    tasks: List[KgTask] = []
    config_catalog: Dict[str, ConfigurationProfile] = {}

    for index, task_key in enumerate(task_name_combo):
        task = task_dict[task_key]
        tasks.append(task)

        if (
            template is not None
            and index < len(template_keys)
            and template_keys[index] == task_key
        ):
            profile = template.config_catalog.get(task.name)
            if profile is not None:
                config_catalog[task.name] = profile
                continue

        if getattr(task, "config_spec", None) is None:
            continue

        bindings: List[ParameterBinding] = []
        name_parts: List[str] = []
        for config_name, config_values in search_space[task_key].items():
            if config_name == "category":
                continue
            if not isinstance(config_values, list):
                raise TypeError(
                    f"Search space values must be lists; got {task_key}.{config_name}={type(config_values)}"
                )
            if not config_values:
                raise ValueError(f"Empty search space for {task_key}.{config_name}")

            config_value = rng.choice(config_values)
            name_parts.append(f"{config_name}={config_value}")
            bindings.append(
                ParameterBinding(
                    parameter=_get_param(task.config_spec, config_name),
                    value=config_value,
                )
            )

        if bindings:
            config_catalog[task.name] = ConfigurationProfile(
                name=f"{task.name}_" + ",".join(name_parts),
                definition=task.config_spec,
                bindings=bindings,
            )

    return PipelineConfig(tasks=tasks, config_catalog=config_catalog)


def print_pipeline_config_short(pipeline_config: PipelineConfig):
    """
    print the pipeline config in a short format
    """
    print()
    print("================")
    for task in pipeline_config.tasks:
        task_name = task.name
        profile: Optional[ConfigurationProfile] = pipeline_config.config_catalog.get(task_name)
        if profile is None:
            print(f"- {task_name}")
            continue

        parts: List[str] = []
        for binding in profile.bindings:
            parts.append(f"{binding.parameter.name}={binding.value}")
        params = ", ".join(parts)
        print(f"- {task_name}({params})")

def sample_config_catalog_for_task_combo(
    search_space: Dict[str, Dict[str, Any]],
    task_name_combo: List[str],
    *,
    rng: random.Random,
) -> PipelineConfig:
    tasks: List[KgTask] = []
    config_catalog: Dict[str, ConfigurationProfile] = {}

    for task_key in task_name_combo:
        task = task_dict[task_key]
        tasks.append(task)

        if getattr(task, "config_spec", None) is None:
            continue

        bindings: List[ParameterBinding] = []
        name_parts: List[str] = []
        for config_name, config_values in search_space[task_key].items():
            if config_name == "category":
                continue
            if not isinstance(config_values, list):
                raise TypeError(
                    f"Search space values must be lists; got {task_key}.{config_name}={type(config_values)}"
                )
            if not config_values:
                raise ValueError(f"Empty search space for {task_key}.{config_name}")

            config_value = rng.choice(config_values)
            name_parts.append(f"{config_name}={config_value}")
            bindings.append(
                ParameterBinding(
                    parameter=_get_param(task.config_spec, config_name),
                    value=config_value,
                )
            )

        if bindings:
            config_catalog[task.name] = ConfigurationProfile(
                name=f"{task.name}_" + ",".join(name_parts),
                definition=task.config_spec,
                bindings=bindings,
            )

    return PipelineConfig(tasks=tasks, config_catalog=config_catalog)

def _task_param_assignments(
    search_space: Dict[str, Dict[str, Any]], task_key: str
) -> List[Dict[str, Any]]:
    space = search_space.get(task_key, {})
    param_space: Dict[str, List[Any]] = {k: v for k, v in space.items() if k != "category"}
    if not param_space:
        return [{}]
    keys = list(param_space.keys())
    values_lists = [param_space[k] for k in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*values_lists)]


def _pipeline_config_for_combo_and_params(
    search_space: Dict[str, Dict[str, Any]],
    combo: List[str],
    assignment_tuple: tuple[Dict[str, Any], ...],
) -> PipelineConfig:
    tasks: List[KgTask] = []
    config_catalog: Dict[str, ConfigurationProfile] = {}

    for task_key, params in zip(combo, assignment_tuple):
        task = task_dict[task_key]
        tasks.append(task)

        if not params:
            continue
        if getattr(task, "config_spec", None) is None:
            continue

        bindings: List[ParameterBinding] = []
        name_parts: List[str] = []

        # Iterate in search_space order for stable snapshots.
        for config_name, _config_values in search_space[task_key].items():
            if config_name == "category":
                continue
            if config_name not in params:
                continue
            config_value = params[config_name]
            name_parts.append(f"{config_name}={config_value}")
            bindings.append(
                ParameterBinding(
                    parameter=_get_param(task.config_spec, config_name),
                    value=config_value,
                )
            )

        config_catalog[task.name] = ConfigurationProfile(
            name=f"{task.name}_" + ",".join(name_parts),
            definition=task.config_spec,
            bindings=bindings,
        )

    return PipelineConfig(tasks=tasks, config_catalog=config_catalog)


def enumerate_snapshots_for_task_combo(
    search_space: Dict[str, Dict[str, Any]],
    combo: List[str],
) -> List[Dict[str, Any]]:
    per_task_assignments = [
        _task_param_assignments(search_space, task_key) for task_key in combo
    ]
    snapshots: List[Dict[str, Any]] = []
    for assignment_tuple in itertools.product(*per_task_assignments):
        pipeline_config = _pipeline_config_for_combo_and_params(
            search_space, combo, assignment_tuple
        )
        snapshots.append(pipeline_config_to_snapshot(combo, pipeline_config))
    return snapshots


def sample_unique_pipeline_config_snapshots_per_combo(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    n: int,
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Sample up to n unique profile snapshots per valid task combo.

    When a combo has fewer than n distinct profile assignments, all available
    profiles are returned for that combo.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    combos = enumerate_valid_task_combinations(search_space, pipeline_layout)
    snapshots: List[Dict[str, Any]] = []
    combo_stats: List[Dict[str, Any]] = []

    for combo in combos:
        available_snapshots = enumerate_snapshots_for_task_combo(search_space, combo)
        serialized = [json.dumps(s, sort_keys=True) for s in available_snapshots]
        if len(set(serialized)) != len(serialized):
            raise ValueError(f"Duplicate profile snapshots for combo {combo!r}")

        sample_count = min(n, len(available_snapshots))
        picked = (
            rng.sample(available_snapshots, k=sample_count)
            if sample_count > 0
            else []
        )
        snapshots.extend(picked)
        combo_stats.append(
            {
                "task_keys": combo,
                "available_profiles": len(available_snapshots),
                "requested": n,
                "sampled": sample_count,
                "exhausted": sample_count < n,
            }
        )

    stats: Dict[str, Any] = {
        "requested_n": n,
        "total_combos": len(combos),
        "total_snapshots": len(snapshots),
        "combos_exhausted": sum(1 for row in combo_stats if row["exhausted"]),
        "combos": combo_stats,
    }
    return snapshots, stats


def enumerate_exhaustive_pipeline_configs(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
) -> List[PipelineConfig]:
    """
    Enumerate every valid pipeline config in the search space.

    Unlike hierarchical sampling (task combo first, then params), this flattens
    the full Cartesian product so each leaf config is equally likely when sampled.
    """
    configs: List[PipelineConfig] = []
    for combo in enumerate_valid_task_combinations(search_space, pipeline_layout):
        per_task_assignments = [
            _task_param_assignments(search_space, task_key) for task_key in combo
        ]
        for assignment_tuple in itertools.product(*per_task_assignments):
            configs.append(
                _pipeline_config_for_combo_and_params(
                    search_space, combo, assignment_tuple
                )
            )
    return configs


def enumerate_exhaustive_pipeline_config_snapshots(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
) -> List[Dict[str, Any]]:
    combos = enumerate_valid_task_combinations(search_space, pipeline_layout)

    all_snapshots: List[Dict[str, Any]] = []
    total_expected = 0

    for combo in combos:
        per_task_assignments = [
            _task_param_assignments(search_space, task_key) for task_key in combo
        ]

        expected_for_combo = 1
        for assignments in per_task_assignments:
            expected_for_combo *= len(assignments)
        total_expected += expected_for_combo

        produced_for_combo = 0
        print()
        print("combo:", combo)
        print("expected configs:", expected_for_combo)

        for assignment_tuple in itertools.product(*per_task_assignments):
            produced_for_combo += 1
            if produced_for_combo % 100 == 1 or produced_for_combo == expected_for_combo:
                print(f"config {produced_for_combo}/{expected_for_combo}")

            pipeline_config = _pipeline_config_for_combo_and_params(
                search_space, combo, assignment_tuple
            )
            all_snapshots.append(pipeline_config_to_snapshot(combo, pipeline_config))

        assert produced_for_combo == expected_for_combo

    print()
    print("TOTAL expected configs:", total_expected)
    print("TOTAL generated snapshots:", len(all_snapshots))
    return all_snapshots
