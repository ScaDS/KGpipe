"""Search-space adapters.

Sampling, category coverage, and snapshot hashing live in
``kgpipe.generation.space``. This module keeps the experiment-facing names and
the paper's ``{task_key: {category, param: values}}`` dicts.
"""

from typing import Any, Dict, List, Optional

from kgpipe.common.model.configuration import ConfigurationProfile
from kgpipe.generation.space import (
    Candidate,
    ConfigurationSpace,
    candidate_from_snapshot,
    candidate_snapshot,
    category_names,
    enumerate_candidates,
    enumerate_task_combinations,
    legacy_snapshot_key,
    lookup_parameter,
    sample_candidate,
    sample_for_combo,
    sample_unique_snapshots,
    snapshots_for_combo,
)
from kgpipe_search.definitions import (
    PipelineConfig,
    RDF_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE,
    RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE,
    RDF_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE,
    TEXT_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE,
    TEXT_SAMPLED_PIPELINE_CONFIGS_FIXTURE,
    TEXT_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE,
    _RDF_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _RDF_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _TEXT_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _TEXT_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    _TEXT_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
    task_dict,
)
import json
import random
from pathlib import Path


def _task_categories_list(search_space: Dict[str, Dict[str, Any]], task_name: str) -> List[str]:
    raw = search_space.get(task_name, {}).get("category")
    return category_names(raw)


def _get_param(definition: Any, param_name: str):
    return lookup_parameter(definition, param_name)


def _space(search_space: Dict[str, Dict[str, Any]], pipeline_layout: Any) -> ConfigurationSpace:
    return ConfigurationSpace.from_legacy(search_space, pipeline_layout, task_dict)


def _as_pipeline_config(candidate: Candidate) -> PipelineConfig:
    return PipelineConfig(tasks=list(candidate.tasks), config_catalog=dict(candidate.profiles))


def enumerate_valid_task_combinations(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: Any,
) -> List[List[str]]:
    """Task-name combinations for the layout, without sampling parameter values."""
    return enumerate_task_combinations(_space(search_space, pipeline_layout))


def pipeline_config_to_snapshot(task_keys: List[str], pipeline_config: PipelineConfig) -> Dict[str, Any]:
    return candidate_snapshot(Candidate(
        list(task_keys),
        list(pipeline_config.tasks),
        dict(pipeline_config.config_catalog),
    ))


def pipeline_config_from_snapshot(snapshot: Dict[str, Any]) -> PipelineConfig:
    return _as_pipeline_config(candidate_from_snapshot(snapshot, task_dict))


def _load_versioned_samples(path: Path, expected_version: int, label: str) -> List[PipelineConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != expected_version:
        raise ValueError(
            f"Unsupported {label} snapshot version {raw.get('version')!r}; "
            f"expected {expected_version}"
        )
    return [pipeline_config_from_snapshot(item) for item in raw["samples"]]


def load_rdf_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    return _load_versioned_samples(
        fixture_path, _RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION, "rdf sampled configs"
    )


def load_text_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or TEXT_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    return _load_versioned_samples(
        fixture_path, _TEXT_PIPELINE_CONFIG_SNAPSHOT_VERSION, "text sampled configs"
    )


def load_rdf_unique_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or RDF_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    return _load_versioned_samples(
        fixture_path,
        _RDF_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
        "rdf unique sampled configs",
    )


def load_text_unique_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or TEXT_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    return _load_versioned_samples(
        fixture_path,
        _TEXT_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
        "text unique sampled configs",
    )


def load_rdf_exhaustive_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or RDF_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE
    return _load_versioned_samples(
        fixture_path,
        _RDF_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
        "rdf exhaustive configs",
    )


def load_text_exhaustive_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or TEXT_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE
    return _load_versioned_samples(
        fixture_path,
        _TEXT_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION,
        "text exhaustive configs",
    )


def sample_valid_pipeline_config(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: Any,
    *,
    rng: Optional[random.Random] = None,
) -> PipelineConfig:
    """Sample one valid pipeline config. The same ``rng`` seed draws the same config."""
    return _as_pipeline_config(sample_candidate(_space(search_space, pipeline_layout), rng))


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
    return legacy_snapshot_key(snapshot)


def build_pipeline_config_for_task_combo(
    search_space: Dict[str, Dict[str, Any]],
    task_name_combo: List[str],
    *,
    rng: random.Random,
    template: Optional[PipelineConfig] = None,
) -> PipelineConfig:
    """Build a pipeline config for a fixed task combo.

    Reuses parameter profiles from ``template`` when the task key is unchanged.
    """
    template_keys = task_keys_from_pipeline_config(template) if template is not None else []
    template_profiles = template.config_catalog if template is not None else {}
    candidate = sample_for_combo(
        _space(search_space, []),
        task_name_combo,
        rng,
        template_keys=template_keys,
        template_profiles=template_profiles,
    )
    return _as_pipeline_config(candidate)


def print_pipeline_config_short(pipeline_config: PipelineConfig):
    """Print the pipeline config in a short format."""
    print()
    print("================")
    for task in pipeline_config.tasks:
        profile: Optional[ConfigurationProfile] = pipeline_config.config_catalog.get(task.name)
        if profile is None:
            print(f"- {task.name}")
            continue
        parts = [f"{binding.parameter.name}={binding.value}" for binding in profile.bindings]
        print(f"- {task.name}({', '.join(parts)})")


def sample_config_catalog_for_task_combo(
    search_space: Dict[str, Dict[str, Any]],
    task_name_combo: List[str],
    *,
    rng: random.Random,
) -> PipelineConfig:
    return build_pipeline_config_for_task_combo(search_space, task_name_combo, rng=rng)


def enumerate_snapshots_for_task_combo(
    search_space: Dict[str, Dict[str, Any]],
    combo: List[str],
) -> List[Dict[str, Any]]:
    return snapshots_for_combo(_space(search_space, []), combo)


def sample_unique_pipeline_config_snapshots_per_combo(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: Any,
    *,
    n: int,
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample up to ``n`` unique profile snapshots per valid task combo."""
    return sample_unique_snapshots(_space(search_space, pipeline_layout), n=n, rng=rng)


def enumerate_exhaustive_pipeline_configs(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: Any,
) -> List[PipelineConfig]:
    """Every valid pipeline config. Each leaf is one task order and one parameter assignment."""
    return [_as_pipeline_config(candidate) for candidate in enumerate_candidates(_space(search_space, pipeline_layout))]


def enumerate_exhaustive_pipeline_config_snapshots(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: Any,
) -> List[Dict[str, Any]]:
    space = _space(search_space, pipeline_layout)
    all_snapshots: List[Dict[str, Any]] = []
    total_expected = 0
    for combo in enumerate_task_combinations(space):
        produced = snapshots_for_combo(space, combo)
        expected_for_combo = len(produced)
        total_expected += expected_for_combo
        print()
        print("combo:", combo)
        print("expected configs:", expected_for_combo)
        for index, snapshot in enumerate(produced, start=1):
            if index % 100 == 1 or index == expected_for_combo:
                print(f"config {index}/{expected_for_combo}")
            all_snapshots.append(snapshot)
    print()
    print("TOTAL expected configs:", total_expected)
    print("TOTAL generated snapshots:", len(all_snapshots))
    return all_snapshots
