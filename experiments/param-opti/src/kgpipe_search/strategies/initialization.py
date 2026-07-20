import random
from typing import Any, Dict, List, Optional, Sequence, Set

from kgpipe_search.configuration import (
    build_pipeline_config_for_task_combo,
    enumerate_valid_task_combinations,
    pipeline_config_snapshot_key,
    sample_valid_pipeline_config,
)
from kgpipe_search.definitions import PipelineConfig, PipelineLayout


def _try_add_unique_config(
    configs: List[PipelineConfig],
    seen: Set[str],
    candidate: PipelineConfig,
    search_space: Dict[str, Dict[str, Any]],
) -> bool:
    key = pipeline_config_snapshot_key(candidate, search_space)
    if key in seen:
        return False
    seen.add(key)
    configs.append(candidate)
    return True


def _fill_unique_configs(
    *,
    configs: List[PipelineConfig],
    seen: Set[str],
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    budget: int,
    rng: random.Random,
    max_attempts_factor: int = 200,
) -> None:
    attempts = 0
    max_attempts = max(1000, max(1, budget - len(configs)) * max_attempts_factor)
    while len(configs) < budget and attempts < max_attempts:
        attempts += 1
        candidate = sample_valid_pipeline_config(search_space, pipeline_layout, rng=rng)
        _try_add_unique_config(configs, seen, candidate, search_space)


def random_initialization(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    budget: int,
    rng: Optional[random.Random] = None,
) -> List[PipelineConfig]:
    """Sample `budget` unique valid pipeline configurations uniformly at random."""
    if budget <= 0:
        return []

    draw = rng or random.Random()
    configs: List[PipelineConfig] = []
    seen: Set[str] = set()
    attempts = 0
    max_attempts = max(1000, budget * 200)

    while len(configs) < budget and attempts < max_attempts:
        attempts += 1
        candidate = sample_valid_pipeline_config(search_space, pipeline_layout, rng=draw)
        key = pipeline_config_snapshot_key(candidate, search_space)
        if key in seen:
            continue
        seen.add(key)
        configs.append(candidate)

    if len(configs) < budget:
        raise RuntimeError(
            f"Failed to sample {budget} unique initial configs (got {len(configs)})."
        )

    return configs


def implementation_aware_initialization(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    budget: int,
    y: int = 1,
    rng: Optional[random.Random] = None,
) -> List[PipelineConfig]:
    """
    Implementation-aware initialization.

    Enumerate (or sample) valid implementation assignments (task combinations) and,
    for each such assignment, generate `y` configurations by sampling parameters.
    """
    if budget <= 0:
        return []
    if y <= 0:
        raise ValueError("y must be >= 1")

    draw = rng or random.Random()
    all_combos = enumerate_valid_task_combinations(search_space, pipeline_layout)
    if not all_combos:
        raise ValueError("No valid implementation assignments found.")

    max_combos = max(1, budget // y)
    combos: Sequence[List[str]]
    if len(all_combos) <= max_combos:
        combos = all_combos
    else:
        combos = draw.sample(all_combos, k=max_combos)

    configs: List[PipelineConfig] = []
    seen: Set[str] = set()

    for combo in combos:
        added_for_combo = 0
        attempts = 0
        max_attempts = max(100, y * 50)
        while (
            added_for_combo < y
            and len(configs) < budget
            and attempts < max_attempts
        ):
            attempts += 1
            candidate = build_pipeline_config_for_task_combo(
                search_space,
                combo,
                rng=draw,
                template=None,
            )
            if _try_add_unique_config(configs, seen, candidate, search_space):
                added_for_combo += 1

        if len(configs) >= budget:
            break

    if len(configs) < budget:
        _fill_unique_configs(
            configs=configs,
            seen=seen,
            search_space=search_space,
            pipeline_layout=pipeline_layout,
            budget=budget,
            rng=draw,
        )

    if len(configs) < budget:
        raise RuntimeError(
            f"Failed to generate {budget} unique initial configs (got {len(configs)}). "
            f"The search space has {len(all_combos)} implementation assignment(s); "
            "try lowering init_budget."
        )

    return configs


