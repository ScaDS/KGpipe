"""
Public API for configuration search.

Implementation lives in `kgpipe_search/strategies/` to keep algorithms modular.
This module preserves the historical function names used by existing tests/scripts.
"""

import random
from typing import Any, Dict

from kgpipe_search.definitions import PipelineLayout
from kgpipe_search.strategies.initialization import (
    implementation_aware_initialization,
    random_initialization,
)
from kgpipe_search.strategies.strategies import (
    EvaluateFn,
    SearchRun,
    run_bayesian,
    run_hnr,
    run_qgns,
    run_random,
)

__all__ = [
    "SearchRun",
    "EvaluateFn",
    "random_initialization",
    "implementation_aware_initialization",
    "random_search",
    "neighborhood_optimization",
    "qgns_search",
    "hnr_search",
    "bayesian_optimization",
]


def random_search(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    rng: Any = None,
) -> SearchRun:
    return run_random(
        budget=budget,
        evaluate_fn=evaluate_fn,
        search_space=search_space,
        pipeline_layout=pipeline_layout,
        rng=rng,
    )


def qgns_search(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    init_budget: int = 0,
    init_strategy: str = "random",
    y: int = 1,
    k: int = 3,
    rho: float = 0.2,
    rng: Any = None,
) -> SearchRun:
    return run_qgns(
        budget=budget,
        evaluate_fn=evaluate_fn,
        search_space=search_space,
        pipeline_layout=pipeline_layout,
        init_budget=init_budget,
        init_strategy="implementation_aware"
        if init_strategy == "implementation_aware"
        else "random",
        y=y,
        k=k,
        rho=rho,
        rng=rng,
    )


def hnr_search(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    init_budget: int,
    init_strategy: str = "implementation_aware",
    y: int = 1,
    rho: float = 0.2,
    rng: Any = None,
) -> SearchRun:
    return run_hnr(
        budget=budget,
        evaluate_fn=evaluate_fn,
        search_space=search_space,
        pipeline_layout=pipeline_layout,
        init_budget=init_budget,
        init_strategy="random" if init_strategy == "random" else "implementation_aware",
        y=y,
        rho=rho,
        rng=rng,
    )


def neighborhood_optimization(
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    k: int = 3,
    rho: float = 0.2,
    rng: Any = None,
    **kwargs: Any,
) -> SearchRun:
    """
    Backwards-compatible alias.

    Historically, this was called `neighborhood_optimization` and implemented QGNS-like behavior.
    """
    del kwargs
    return qgns_search(
        budget=budget,
        evaluate_fn=evaluate_fn,
        search_space=search_space,
        pipeline_layout=pipeline_layout,
        init_budget=0,
        init_strategy="random",
        k=k,
        rho=rho,
        rng=rng,
    )


def bayesian_optimization(
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    *,
    init_random: int = 3,
    init_strategy: str = "random",
    y: int = 1,
    pool_size: int = 32,
    beta: float = 0.5,
    rng: Any = None,
    **kwargs: Any,
) -> SearchRun:
    del kwargs
    return run_bayesian(
        budget=budget,
        evaluate_fn=evaluate_fn,
        search_space=search_space,
        pipeline_layout=pipeline_layout,
        init_budget=init_random,
        init_strategy="implementation_aware"
        if init_strategy == "implementation_aware"
        else "random",
        y=y,
        pool_size=pool_size,
        beta=beta,
        rng=rng,
    )