import os
import random
from pathlib import Path

import pytest

from kgpipe.common import Data, DataFormat, KgPipe
from kgpipe_search.configuration import print_pipeline_config_short
from kgpipe_search.definitions import RDF_PIPELINE_LAYOUT, RDF_SEARCH_SPACE, PipelineConfig
from kgpipe_search.evaluation import evaluate_pipeline
from kgpipe_search.search import random_search

BUDGET = 10
SEED = 42

ONTOLOGY_PATH = Path("data/bench/moviekg_datasets/film_10k/ontology.ttl")
SEED_PATH = Path("data/bench/moviekg_datasets/film_10k/split_0/kg/seed/data.nt")
REFERENCE_PATH = Path("data/bench/moviekg_datasets/film_10k/split_1/kg/reference/data_agg.nt")
RDF_SOURCE_PATH = Path("data/bench/moviekg_datasets/film_10k/split_0/sources/rdf/data.nt")
RDF_TMP_DIR = Path("data/tmp/rdf_pipelines")


def _bench_dataset_available() -> bool:
    return all(
        path.exists()
        for path in (ONTOLOGY_PATH, SEED_PATH, REFERENCE_PATH, RDF_SOURCE_PATH)
    )


def _run_rdf_pipeline(
    pipeline_config: PipelineConfig,
    *,
    result_path: Path,
    tasks_tmp_dir: Path,
    run_name: str,
) -> Path:
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = KgPipe(
        tasks=pipeline_config.tasks,
        seed=Data(path=SEED_PATH, format=DataFormat.RDF_NTRIPLES),
        data_dir=tasks_tmp_dir,
        name=run_name,
    )

    pipeline.build(
        stable_files=True,
        configCatalog=pipeline_config.config_catalog,
        source=Data(path=RDF_SOURCE_PATH, format=DataFormat.RDF_NTRIPLES),
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES),
    )

    pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=False)
    return result_path


def test_rdf_pipeline_random_search():
    if not _bench_dataset_available():
        pytest.skip("moviekg bench dataset not available under data/bench/moviekg_datasets/film_10k")

    os.environ["ONTOLOGY_PATH"] = str(ONTOLOGY_PATH)
    RDF_TMP_DIR.mkdir(parents=True, exist_ok=True)

    trial_counter = {"n": 0}

    def evaluate_fn(pipeline_config: PipelineConfig) -> float:
        trial = trial_counter["n"]
        trial_counter["n"] += 1

        result_path = RDF_TMP_DIR / f"random_search_trial_{trial}.nt"
        tasks_tmp_dir = RDF_TMP_DIR / f"random_search_trial_{trial}_tasks_tmp"

        _run_rdf_pipeline(
            pipeline_config,
            result_path=result_path,
            tasks_tmp_dir=tasks_tmp_dir,
            run_name=f"random_search_trial_{trial}",
        )

        aggregate_score = evaluate_pipeline(
            pipeline_config,
            result_path,
            REFERENCE_PATH,
        )
        return aggregate_score.final_score

    run = random_search(
        budget=BUDGET,
        evaluate_fn=evaluate_fn,
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        rng=random.Random(SEED),
    )

    print("\n=== rdf pipeline random search ===")
    print(f"budget: {BUDGET}")
    print(f"seed: {SEED}")

    best_score = float("-inf")
    for trial, ((score, pipeline_config), decision) in enumerate(
        zip(run.history, run.decisions),
        start=1,
    ):
        if score > best_score:
            best_score = score
            improved = " (new best)"
        else:
            improved = ""

        print(f"\n--- trial {trial}/{BUDGET} [{decision}] ---")
        print_pipeline_config_short(pipeline_config)
        print(f"score: {score:.4f}{improved}")
        print(f"best so far: {best_score:.4f}")

    assert len(run.history) == BUDGET
    assert len(run.decisions) == BUDGET
    assert best_score > float("-inf")
