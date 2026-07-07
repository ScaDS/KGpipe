from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

from kgpipe.common.model.configuration import ConfigurationProfile, ParameterBinding
from kgpipe_search.configuration import (
    build_pipeline_config_for_task_combo,
    enumerate_valid_task_combinations,
    pipeline_config_snapshot_key,
    sample_valid_pipeline_config,
    task_keys_from_pipeline_config,
)
from kgpipe_search.definitions import PipelineConfig, PipelineLayout

Observation = Tuple[float, PipelineConfig]
EvaluateFn = Callable[[PipelineConfig], float]
SearchStrategy = Literal["random", "neighborhood", "bayesian"]


@dataclass
class SearchRun:
    strategy: SearchStrategy
    history: List[Observation]
    budget: int
    decisions: List[str]


def _top_k(history: List[Observation], k: int) -> List[Observation]:
    ranked = sorted(history, key=lambda item: item[0], reverse=True)
    return ranked[: max(1, min(k, len(ranked)))]


def _parameter_neighbors(
    anchor: PipelineConfig,
    search_space: Dict[str, Dict[str, Any]],
) -> List[PipelineConfig]:
    anchor_keys = task_keys_from_pipeline_config(anchor)
    neighbors: List[PipelineConfig] = []

    for task, task_key in zip(anchor.tasks, anchor_keys):
        profile = anchor.config_catalog.get(task.name)
        if profile is None:
            continue

        for binding in profile.bindings:
            param_name = binding.parameter.name
            domain = search_space.get(task_key, {}).get(param_name)
            if not isinstance(domain, list):
                continue

            for value in domain:
                if value == binding.value:
                    continue

                new_catalog = dict(anchor.config_catalog)
                new_bindings: List[ParameterBinding] = []
                name_parts: List[str] = []
                for current in profile.bindings:
                    chosen = value if current.parameter.name == param_name else current.value
                    new_bindings.append(
                        ParameterBinding(parameter=current.parameter, value=chosen)
                    )
                    name_parts.append(f"{current.parameter.name}={chosen}")

                new_catalog[task.name] = ConfigurationProfile(
                    name=f"{task.name}_" + ",".join(name_parts),
                    definition=profile.definition,
                    bindings=new_bindings,
                )
                neighbors.append(
                    PipelineConfig(tasks=list(anchor.tasks), config_catalog=new_catalog)
                )

    return neighbors


def _implementation_neighbors(
    anchor: PipelineConfig,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    rng: random.Random,
) -> List[PipelineConfig]:
    anchor_keys = task_keys_from_pipeline_config(anchor)
    neighbors: List[PipelineConfig] = []

    for combo in enumerate_valid_task_combinations(search_space, pipeline_layout):
        if len(combo) != len(anchor_keys):
            continue
        if sum(left != right for left, right in zip(anchor_keys, combo)) != 1:
            continue
        neighbors.append(
            build_pipeline_config_for_task_combo(
                search_space,
                combo,
                rng=rng,
                template=anchor,
            )
        )

    return neighbors


def neighbors_at_distance_one(
    anchor: PipelineConfig,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    rng: random.Random,
) -> List[PipelineConfig]:
    seen: Set[str] = set()
    neighbors: List[PipelineConfig] = []

    for candidate in (
        _parameter_neighbors(anchor, search_space)
        + _implementation_neighbors(anchor, search_space, pipeline_layout, rng)
    ):
        key = pipeline_config_snapshot_key(candidate, search_space)
        if key in seen:
            continue
        seen.add(key)
        neighbors.append(candidate)

    return neighbors


def sample_unevaluated_config(
    rng: random.Random,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    evaluated_keys: Set[str],
    *,
    max_attempts: int = 500,
) -> PipelineConfig:
    for _ in range(max_attempts):
        candidate = sample_valid_pipeline_config(
            search_space,
            pipeline_layout,
            rng=rng,
        )
        key = pipeline_config_snapshot_key(candidate, search_space)
        if key not in evaluated_keys:
            return candidate

    raise RuntimeError("Failed to sample an unevaluated configuration")


def select_next_random_config(
    rng: random.Random,
    history: List[Observation],
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    evaluated_keys: Set[str],
) -> PipelineConfig:
    del history
    return sample_unevaluated_config(
        rng,
        search_space,
        pipeline_layout,
        evaluated_keys,
    )


def select_next_neighborhood_config(
    rng: random.Random,
    history: List[Observation],
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    evaluated_keys: Set[str],
    *,
    k: int = 3,
    rho: float = 0.2,
) -> Tuple[PipelineConfig, str]:
    if not history or rng.random() < rho:
        return (
            sample_unevaluated_config(
                rng,
                search_space,
                pipeline_layout,
                evaluated_keys,
            ),
            "explore",
        )

    anchors = _top_k(history, k)
    anchor_score, anchor_config = rng.choice(anchors)
    neighborhood = neighbors_at_distance_one(
        anchor_config,
        search_space,
        pipeline_layout,
        rng,
    )

    unevaluated = [
        candidate
        for candidate in neighborhood
        if pipeline_config_snapshot_key(candidate, search_space) not in evaluated_keys
    ]
    if unevaluated:
        return rng.choice(unevaluated), f"neighborhood(anchor_score={anchor_score:.4f})"

    return (
        sample_unevaluated_config(
            rng,
            search_space,
            pipeline_layout,
            evaluated_keys,
        ),
        "explore(fallback)",
    )


def _config_distance(
    left: PipelineConfig,
    right: PipelineConfig,
    search_space: Dict[str, Dict[str, Any]],
) -> float:
    if pipeline_config_snapshot_key(left, search_space) == pipeline_config_snapshot_key(
        right, search_space
    ):
        return 0.0

    left_keys = task_keys_from_pipeline_config(left)
    right_keys = task_keys_from_pipeline_config(right)
    distance = float(sum(a != b for a, b in zip(left_keys, right_keys)))
    if len(left_keys) != len(right_keys):
        distance += abs(len(left_keys) - len(right_keys))

    left_params = {
        (task.name, binding.parameter.name): binding.value
        for task in left.tasks
        for binding in (left.config_catalog.get(task.name).bindings if left.config_catalog.get(task.name) else [])
    }
    right_params = {
        (task.name, binding.parameter.name): binding.value
        for task in right.tasks
        for binding in (right.config_catalog.get(task.name).bindings if right.config_catalog.get(task.name) else [])
    }

    all_param_keys = set(left_params) | set(right_params)
    for key in all_param_keys:
        if left_params.get(key) != right_params.get(key):
            distance += 1.0

    return distance


def _predict_with_uncertainty(
    candidate: PipelineConfig,
    history: List[Observation],
    search_space: Dict[str, Dict[str, Any]],
) -> Tuple[float, float]:
    weights: List[float] = []
    scores: List[float] = []

    for score, observed in history:
        distance = _config_distance(candidate, observed, search_space)
        if distance == 0.0:
            return score, 0.0
        weights.append(math.exp(-distance))
        scores.append(score)

    if not weights:
        return 0.75, 1.0

    total_weight = sum(weights)
    mean = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
    uncertainty = 1.0 / (1.0 + total_weight)
    return mean, uncertainty


def _acquisition(mean: float, uncertainty: float, *, beta: float = 0.5) -> float:
    return mean + beta * uncertainty


def select_next_bayesian_config(
    rng: random.Random,
    history: List[Observation],
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    evaluated_keys: Set[str],
    *,
    init_random: int = 3,
    pool_size: int = 32,
    beta: float = 0.5,
) -> Tuple[PipelineConfig, str]:
    if len(history) < init_random:
        return (
            sample_unevaluated_config(
                rng,
                search_space,
                pipeline_layout,
                evaluated_keys,
            ),
            "init_random",
        )

    candidates: List[PipelineConfig] = []
    for _ in range(pool_size):
        candidates.append(
            sample_unevaluated_config(
                rng,
                search_space,
                pipeline_layout,
                evaluated_keys,
            )
        )

    best_candidate = candidates[0]
    best_acquisition = float("-inf")
    best_prediction = 0.0
    best_uncertainty = 0.0

    for candidate in candidates:
        mean, uncertainty = _predict_with_uncertainty(candidate, history, search_space)
        score = _acquisition(mean, uncertainty, beta=beta)
        if score > best_acquisition:
            best_acquisition = score
            best_candidate = candidate
            best_prediction = mean
            best_uncertainty = uncertainty

    return (
        best_candidate,
        f"acquisition(pred={best_prediction:.4f},unc={best_uncertainty:.4f},a={best_acquisition:.4f})",
    )


def run_search(
    strategy: SearchStrategy,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    rng: Optional[random.Random] = None,
    k: int = 3,
    rho: float = 0.2,
    init_random: int = 3,
    pool_size: int = 32,
    beta: float = 0.5,
) -> SearchRun:
    draw = rng or random.Random()
    history: List[Observation] = []
    evaluated_keys: Set[str] = set()
    decisions: List[str] = []

    for _ in range(budget):
        if strategy == "random":
            candidate = select_next_random_config(
                draw,
                history,
                search_space,
                pipeline_layout,
                evaluated_keys,
            )
            decision = "sample"
        elif strategy == "neighborhood":
            candidate, decision = select_next_neighborhood_config(
                draw,
                history,
                search_space,
                pipeline_layout,
                evaluated_keys,
                k=k,
                rho=rho,
            )
        elif strategy == "bayesian":
            candidate, decision = select_next_bayesian_config(
                draw,
                history,
                search_space,
                pipeline_layout,
                evaluated_keys,
                init_random=init_random,
                pool_size=pool_size,
                beta=beta,
            )
        else:
            raise ValueError(f"Unknown search strategy: {strategy!r}")

        key = pipeline_config_snapshot_key(candidate, search_space)
        score = evaluate_fn(candidate)
        history.append((score, candidate))
        evaluated_keys.add(key)
        decisions.append(decision)

    return SearchRun(
        strategy=strategy,
        history=history,
        budget=budget,
        decisions=decisions,
    )


def random_search(
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    **kwargs: Any,
) -> SearchRun:
    return run_search(
        "random",
        budget,
        evaluate_fn,
        search_space,
        pipeline_layout,
        **kwargs,
    )


def neighborhood_optimization(
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    k: int = 3,
    rho: float = 0.2,
    **kwargs: Any,
) -> SearchRun:
    return run_search(
        "neighborhood",
        budget,
        evaluate_fn,
        search_space,
        pipeline_layout,
        k=k,
        rho=rho,
        **kwargs,
    )


def bayesian_optimization(
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    init_random: int = 3,
    pool_size: int = 32,
    beta: float = 0.5,
    **kwargs: Any,
) -> SearchRun:
    return run_search(
        "bayesian",
        budget,
        evaluate_fn,
        search_space,
        pipeline_layout,
        init_random=init_random,
        pool_size=pool_size,
        beta=beta,
        **kwargs,
    )
