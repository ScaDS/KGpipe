import random

from kgpipe_search.configuration import enumerate_valid_task_combinations
from kgpipe_search.definitions import RDF_PIPELINE_LAYOUT, RDF_SEARCH_SPACE
from kgpipe_search.evaluation import dummy_evaluate_pipeline
from kgpipe_search.search import hnr_search, implementation_aware_search, qgns_search
from kgpipe_search.strategies.initialization import implementation_aware_initialization


def _assert_valid(run) -> None:
    assert len(run.history) == run.budget
    assert len(run.decisions) == run.budget

    seen: set[str] = set()
    for score, cfg in run.history:
        assert 0.5 <= score <= 1.0
        assert cfg.tasks
        # Snapshot key uniqueness is the true criterion; repr is good enough here.
        key = repr(
            [
                (
                    task.name,
                    tuple(
                        (b.parameter.name, b.value)
                        for b in (
                            cfg.config_catalog.get(task.name).bindings
                            if cfg.config_catalog.get(task.name)
                            else []
                        )
                    ),
                )
                for task in cfg.tasks
            ]
        )
        assert key not in seen
        seen.add(key)


def test_dummy_evaluate_pipeline_qgns_with_implementation_aware_init():
    run = qgns_search(
        budget=10,
        init_budget=3,
        init_strategy="implementation_aware",
        y=1,
        evaluate_fn=dummy_evaluate_pipeline,
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        k=3,
        rho=0.2,
        rng=random.Random(3),
    )
    _assert_valid(run)


def test_dummy_evaluate_pipeline_hnr():
    run = hnr_search(
        budget=10,
        init_budget=4,
        init_strategy="implementation_aware",
        y=1,
        evaluate_fn=dummy_evaluate_pipeline,
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        rho=0.2,
        rng=random.Random(4),
    )
    _assert_valid(run)


def test_dummy_evaluate_pipeline_implementation_aware_search():
    run = implementation_aware_search(
        budget=10,
        y=1,
        evaluate_fn=dummy_evaluate_pipeline,
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        rng=random.Random(5),
    )
    _assert_valid(run)
    assert any(str(d).startswith("init(implementation_aware)") for d in run.decisions)


def test_implementation_aware_init_can_exceed_task_combo_count():
    combos = enumerate_valid_task_combinations(RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT)
    init_budget = len(combos) + 1

    configs = implementation_aware_initialization(
        RDF_SEARCH_SPACE,
        RDF_PIPELINE_LAYOUT,
        budget=init_budget,
        y=1,
        rng=random.Random(0),
    )
    assert len(configs) == init_budget

