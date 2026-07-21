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
from kgpipe_search.strategies.initialization import (
    implementation_aware_initialization,
    random_initialization,
)

Observation = Tuple[float, PipelineConfig]
EvaluateFn = Callable[[PipelineConfig], float]

SearchStrategy = Literal["random", "implementation_aware", "qgns", "hnr", "bayesian", "llm"]


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


def _restricted_implementation_neighbors_for_index(
    anchor: PipelineConfig,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    rng: random.Random,
    *,
    index: int,
) -> List[PipelineConfig]:
    anchor_keys = task_keys_from_pipeline_config(anchor)
    if index < 0 or index >= len(anchor_keys):
        return []

    neighbors: List[PipelineConfig] = []
    for combo in enumerate_valid_task_combinations(search_space, pipeline_layout):
        if len(combo) != len(anchor_keys):
            continue
        if any(i != index and combo[i] != anchor_keys[i] for i in range(len(anchor_keys))):
            continue
        if combo[index] == anchor_keys[index]:
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


def _restricted_parameter_neighbors_for_index(
    anchor: PipelineConfig,
    search_space: Dict[str, Dict[str, Any]],
    *,
    index: int,
) -> List[PipelineConfig]:
    anchor_keys = task_keys_from_pipeline_config(anchor)
    if index < 0 or index >= len(anchor.tasks) or index >= len(anchor_keys):
        return []

    task = anchor.tasks[index]
    task_key = anchor_keys[index]
    profile = anchor.config_catalog.get(task.name)
    if profile is None:
        return []

    neighbors: List[PipelineConfig] = []
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
        for binding in (
            left.config_catalog.get(task.name).bindings
            if left.config_catalog.get(task.name)
            else []
        )
    }
    right_params = {
        (task.name, binding.parameter.name): binding.value
        for task in right.tasks
        for binding in (
            right.config_catalog.get(task.name).bindings
            if right.config_catalog.get(task.name)
            else []
        )
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


def run_random(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    rng: Optional[random.Random] = None,
) -> SearchRun:
    draw = rng or random.Random()
    history: List[Observation] = []
    decisions: List[str] = []
    evaluated_keys: Set[str] = set()

    for _ in range(budget):
        candidate = sample_unevaluated_config(
            draw, search_space, pipeline_layout, evaluated_keys
        )
        key = pipeline_config_snapshot_key(candidate, search_space)
        score = evaluate_fn(candidate)
        history.append((score, candidate))
        evaluated_keys.add(key)
        decisions.append("sample")

    return SearchRun(strategy="random", history=history, budget=budget, decisions=decisions)


def run_implementation_aware(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    y: int = 1,
    rng: Optional[random.Random] = None,
) -> SearchRun:
    """
    Evaluate `budget` configs from implementation-aware initialization.

    Task combinations are covered systematically (`y` random parameter samples per combo).
    Any remaining budget is filled with uniform random valid configs.
    """
    if budget <= 0:
        return SearchRun(
            strategy="implementation_aware",
            history=[],
            budget=0,
            decisions=[],
        )

    draw = rng or random.Random()
    history: List[Observation] = []
    decisions: List[str] = []
    evaluated_keys: Set[str] = set()

    init_set = implementation_aware_initialization(
        search_space,
        pipeline_layout,
        budget=budget,
        y=y,
        rng=draw,
    )

    for cfg in init_set:
        if len(history) >= budget:
            break
        key = pipeline_config_snapshot_key(cfg, search_space)
        if key in evaluated_keys:
            continue
        score = evaluate_fn(cfg)
        history.append((score, cfg))
        evaluated_keys.add(key)
        decisions.append("init(implementation_aware)")

    while len(history) < budget:
        candidate = sample_unevaluated_config(
            draw, search_space, pipeline_layout, evaluated_keys
        )
        key = pipeline_config_snapshot_key(candidate, search_space)
        score = evaluate_fn(candidate)
        history.append((score, candidate))
        evaluated_keys.add(key)
        decisions.append("sample")

    return SearchRun(
        strategy="implementation_aware",
        history=history,
        budget=budget,
        decisions=decisions,
    )


def run_qgns(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    init_budget: int = 0,
    init_strategy: Literal["random", "implementation_aware"] = "random",
    y: int = 1,
    k: int = 3,
    rho: float = 0.2,
    rng: Optional[random.Random] = None,
) -> SearchRun:
    if budget <= 0:
        return SearchRun(strategy="qgns", history=[], budget=0, decisions=[])

    draw = rng or random.Random()
    history: List[Observation] = []
    decisions: List[str] = []
    evaluated_keys: Set[str] = set()

    if init_budget > 0:
        if init_strategy == "implementation_aware":
            init_set = implementation_aware_initialization(
                search_space,
                pipeline_layout,
                budget=min(init_budget, budget),
                y=y,
                rng=draw,
            )
        else:
            init_set = random_initialization(
                search_space,
                pipeline_layout,
                budget=min(init_budget, budget),
                rng=draw,
            )
        for cfg in init_set:
            key = pipeline_config_snapshot_key(cfg, search_space)
            if key in evaluated_keys:
                continue
            score = evaluate_fn(cfg)
            history.append((score, cfg))
            evaluated_keys.add(key)
            decisions.append(f"init({init_strategy})")
            if len(history) >= budget:
                return SearchRun(strategy="qgns", history=history, budget=budget, decisions=decisions)

    while len(history) < budget:
        if not history or draw.random() < rho:
            candidate = sample_unevaluated_config(
                draw, search_space, pipeline_layout, evaluated_keys
            )
            decision = "explore"
        else:
            anchors = _top_k(history, k)
            candidate = None
            decision = "explore(fallback)"

            shuffled = list(anchors)
            draw.shuffle(shuffled)
            for anchor_score, anchor_cfg in shuffled:
                neighborhood = neighbors_at_distance_one(
                    anchor_cfg, search_space, pipeline_layout, draw
                )
                unevaluated = [
                    n
                    for n in neighborhood
                    if pipeline_config_snapshot_key(n, search_space) not in evaluated_keys
                ]
                if not unevaluated:
                    continue
                candidate = draw.choice(unevaluated)
                decision = f"neighborhood(anchor_score={anchor_score:.4f})"
                break

            if candidate is None:
                candidate = sample_unevaluated_config(
                    draw, search_space, pipeline_layout, evaluated_keys
                )

        key = pipeline_config_snapshot_key(candidate, search_space)
        score = evaluate_fn(candidate)
        history.append((score, candidate))
        evaluated_keys.add(key)
        decisions.append(decision)

    return SearchRun(strategy="qgns", history=history, budget=budget, decisions=decisions)


def run_hnr(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    init_budget: int,
    init_strategy: Literal["random", "implementation_aware"] = "implementation_aware",
    y: int = 1,
    rho: float = 0.2,
    rng: Optional[random.Random] = None,
) -> SearchRun:
    if budget <= 0:
        return SearchRun(strategy="hnr", history=[], budget=0, decisions=[])
    if init_budget <= 0:
        raise ValueError("HNR requires init_budget > 0")

    draw = rng or random.Random()
    history: List[Observation] = []
    decisions: List[str] = []
    evaluated_keys: Set[str] = set()

    if init_strategy == "implementation_aware":
        init_set = implementation_aware_initialization(
            search_space,
            pipeline_layout,
            budget=min(init_budget, budget),
            y=y,
            rng=draw,
        )
    else:
        init_set = random_initialization(
            search_space,
            pipeline_layout,
            budget=min(init_budget, budget),
            rng=draw,
        )

    for cfg in init_set:
        key = pipeline_config_snapshot_key(cfg, search_space)
        if key in evaluated_keys:
            continue
        score = evaluate_fn(cfg)
        history.append((score, cfg))
        evaluated_keys.add(key)
        decisions.append(f"init({init_strategy})")
        if len(history) >= budget:
            return SearchRun(strategy="hnr", history=history, budget=budget, decisions=decisions)

    best_score, best_cfg = max(history, key=lambda item: item[0])

    while len(history) < budget:
        improved = False

        for idx in range(len(best_cfg.tasks)):
            if len(history) >= budget:
                break

            if draw.random() < rho:
                candidate = sample_unevaluated_config(
                    draw, search_space, pipeline_layout, evaluated_keys
                )
                decision = f"explore(task_idx={idx})"
            else:
                task_neighbors = _restricted_implementation_neighbors_for_index(
                    best_cfg, search_space, pipeline_layout, draw, index=idx
                )
                task_candidates = [
                    n
                    for n in task_neighbors
                    if pipeline_config_snapshot_key(n, search_space) not in evaluated_keys
                ]

                if task_candidates:
                    candidate = draw.choice(task_candidates)
                    decision = f"task_neighbor(idx={idx})"
                else:
                    param_neighbors = _restricted_parameter_neighbors_for_index(
                        best_cfg, search_space, index=idx
                    )
                    param_candidates = [
                        n
                        for n in param_neighbors
                        if pipeline_config_snapshot_key(n, search_space) not in evaluated_keys
                    ]
                    if param_candidates:
                        candidate = draw.choice(param_candidates)
                        decision = f"param_neighbor(idx={idx})"
                    else:
                        candidate = sample_unevaluated_config(
                            draw, search_space, pipeline_layout, evaluated_keys
                        )
                        decision = f"explore(fallback,idx={idx})"

            key = pipeline_config_snapshot_key(candidate, search_space)
            score = evaluate_fn(candidate)
            history.append((score, candidate))
            evaluated_keys.add(key)
            decisions.append(decision)

            if score > best_score:
                best_score, best_cfg = score, candidate
                improved = True

        if not improved and len(history) < budget and draw.random() < rho:
            candidate = sample_unevaluated_config(
                draw, search_space, pipeline_layout, evaluated_keys
            )
            key = pipeline_config_snapshot_key(candidate, search_space)
            score = evaluate_fn(candidate)
            history.append((score, candidate))
            evaluated_keys.add(key)
            decisions.append("explore(post_sweep)")
            if score > best_score:
                best_score, best_cfg = score, candidate

    return SearchRun(strategy="hnr", history=history, budget=budget, decisions=decisions)


def run_bayesian(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    init_budget: int = 3,
    init_strategy: Literal["random", "implementation_aware"] = "random",
    y: int = 1,
    pool_size: int = 32,
    beta: float = 0.5,
    rng: Optional[random.Random] = None,
) -> SearchRun:
    if budget <= 0:
        return SearchRun(strategy="bayesian", history=[], budget=0, decisions=[])

    draw = rng or random.Random()
    history: List[Observation] = []
    decisions: List[str] = []
    evaluated_keys: Set[str] = set()

    if init_budget > 0:
        if init_strategy == "implementation_aware":
            init_set = implementation_aware_initialization(
                search_space,
                pipeline_layout,
                budget=min(init_budget, budget),
                y=y,
                rng=draw,
            )
        else:
            init_set = random_initialization(
                search_space,
                pipeline_layout,
                budget=min(init_budget, budget),
                rng=draw,
            )
        for cfg in init_set:
            key = pipeline_config_snapshot_key(cfg, search_space)
            if key in evaluated_keys:
                continue
            score = evaluate_fn(cfg)
            history.append((score, cfg))
            evaluated_keys.add(key)
            decisions.append(f"init({init_strategy})")
            if len(history) >= budget:
                return SearchRun(strategy="bayesian", history=history, budget=budget, decisions=decisions)

    while len(history) < budget:
        candidates: List[PipelineConfig] = []
        for _ in range(pool_size):
            candidates.append(
                sample_unevaluated_config(
                    draw, search_space, pipeline_layout, evaluated_keys
                )
            )

        best_candidate = candidates[0]
        best_acq = float("-inf")
        best_pred = 0.0
        best_unc = 0.0

        for candidate in candidates:
            mean, unc = _predict_with_uncertainty(candidate, history, search_space)
            acq = _acquisition(mean, unc, beta=beta)
            if acq > best_acq:
                best_acq = acq
                best_candidate = candidate
                best_pred = mean
                best_unc = unc

        key = pipeline_config_snapshot_key(best_candidate, search_space)
        score = evaluate_fn(best_candidate)
        history.append((score, best_candidate))
        evaluated_keys.add(key)
        decisions.append(
            f"acquisition(pred={best_pred:.4f},unc={best_unc:.4f},a={best_acq:.4f})"
        )

    return SearchRun(strategy="bayesian", history=history, budget=budget, decisions=decisions)

