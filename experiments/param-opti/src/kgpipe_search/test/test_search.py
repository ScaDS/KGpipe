import random

from kgpipe_search.configuration import print_pipeline_config_short
from kgpipe_search.definitions import RDF_PIPELINE_LAYOUT, RDF_SEARCH_SPACE, PipelineConfig
from kgpipe_search.evaluation import dummy_evaluate_pipeline
from kgpipe_search.search import (
    SearchRun,
    bayesian_optimization,
    neighborhood_optimization,
    random_search,
)


def _print_search_path(run: SearchRun, pipeline_layout) -> None:
    best_score = float("-inf")
    best_config: PipelineConfig | None = None

    print(f"\n=== {run.strategy} search ===")
    print(f"budget: {run.budget}")
    print(f"layout: {pipeline_layout.allowed_task_categories}")

    for trial, ((score, pipeline_config), decision) in enumerate(
        zip(run.history, run.decisions),
        start=1,
    ):
        print(f"\n--- trial {trial}/{run.budget} [{decision}] ---")
        print_pipeline_config_short(pipeline_config)

        if score > best_score:
            best_score = score
            best_config = pipeline_config
            improved = " (new best)"
        else:
            improved = ""

        print(f"score: {score:.4f}{improved}")
        print(f"best so far: {best_score:.4f}")

    print("\n=== search summary ===")
    print(f"evaluated: {len(run.history)}")
    print(f"best score: {best_score:.4f}")
    if best_config is not None:
        print("best config:")
        print_pipeline_config_short(best_config)


def _assert_valid_search_run(run: SearchRun) -> None:
    assert len(run.history) == run.budget
    assert len(run.decisions) == run.budget

    seen_configs: set[str] = set()
    for score, pipeline_config in run.history:
        assert 0.5 <= score <= 1.0
        assert pipeline_config.tasks
        config_repr = repr(
            [
                (
                    task.name,
                    tuple(
                        (binding.parameter.name, binding.value)
                        for binding in (
                            pipeline_config.config_catalog.get(task.name).bindings
                            if pipeline_config.config_catalog.get(task.name)
                            else []
                        )
                    ),
                )
                for task in pipeline_config.tasks
            ]
        )
        assert config_repr not in seen_configs
        seen_configs.add(config_repr)


def test_dummy_evaluate_pipeline_random_search_strategy():
    run = random_search(
        budget=10,
        evaluate_fn=dummy_evaluate_pipeline,
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        rng=random.Random(0),
    )
    _print_search_path(run, RDF_PIPELINE_LAYOUT)
    _assert_valid_search_run(run)


def test_dummy_evaluate_pipeline_neighborhood_search_strategy():
    run = neighborhood_optimization(
        budget=10,
        evaluate_fn=dummy_evaluate_pipeline,
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        k=3,
        rho=0.2,
        rng=random.Random(1),
    )
    _print_search_path(run, RDF_PIPELINE_LAYOUT)
    _assert_valid_search_run(run)


def test_dummy_evaluate_pipeline_bayesian_search_strategy():
    run = bayesian_optimization(
        budget=10,
        evaluate_fn=dummy_evaluate_pipeline,
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        init_random=3,
        pool_size=16,
        rng=random.Random(2),
    )
    _print_search_path(run, RDF_PIPELINE_LAYOUT)
    _assert_valid_search_run(run)
