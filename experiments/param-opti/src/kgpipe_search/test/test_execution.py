import os
import random
from pathlib import Path

import pytest

from kgpipe.common import Data, DataFormat, KgPipe
from kgpipe_search.configuration import (
    load_pipeline_config_snapshot,
    load_rdf_sampled_pipeline_configs,
    pipeline_config_snapshot_key,
    save_pipeline_config_snapshot,
    sample_valid_pipeline_config,
)
from kgpipe_search.definitions import RDF_PIPELINE_LAYOUT, RDF_SEARCH_SPACE

KGPIPE_ROOT = Path(__file__).resolve().parents[5]
FALLBACK_TEST_DATA = KGPIPE_ROOT / "src/kgpipe_tasks/test/test_data/rdf"

tmp_base_dir = Path("data/tmp/rdf_pipelines")
tmp_base_dir.mkdir(parents=True, exist_ok=True)

SEED_PATH = Path("data/input_final/target_kg/graph.nt")
SOURCE_PATH = Path("data/input_final/rdf_source/graph.nt")
ONTOLOGY_PATH = Path("data/input_final/target_kg/ontology.ttl")
FALLBACK_SEED_PATH = FALLBACK_TEST_DATA / "target.nt"
FALLBACK_SOURCE_PATH = FALLBACK_TEST_DATA / "source.nt"
FALLBACK_ONTOLOGY_PATH = FALLBACK_TEST_DATA / "ontology.ttl"


def _ensure_ontology_env() -> None:
    if ONTOLOGY_PATH.exists():
        os.environ["ONTOLOGY_PATH"] = str(ONTOLOGY_PATH)
    elif FALLBACK_ONTOLOGY_PATH.exists():
        os.environ["ONTOLOGY_PATH"] = str(FALLBACK_ONTOLOGY_PATH)


def _rdf_input_paths(tmp_dir: Path) -> tuple[Path, Path]:
    if SEED_PATH.exists() and SOURCE_PATH.exists():
        return SEED_PATH, SOURCE_PATH
    if FALLBACK_SEED_PATH.exists() and FALLBACK_SOURCE_PATH.exists():
        return FALLBACK_SEED_PATH, FALLBACK_SOURCE_PATH

    seed_path = tmp_dir / "seed.nt"
    source_path = tmp_dir / "source.nt"
    seed_path.write_text(
        "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n",
        encoding="utf-8",
    )
    source_path.write_text(
        "<http://example.org/s2> <http://example.org/p> <http://example.org/o> .\n",
        encoding="utf-8",
    )
    return seed_path, source_path


def _run_rdf_pipeline_config(
    pipeline_config,
    *,
    tmp_dir: Path,
    run_name: str,
    result_path: Path,
) -> Path:
    _ensure_ontology_env()
    seed_path, source_path = _rdf_input_paths(tmp_dir)
    tasks_tmp_dir = tmp_dir / f"{run_name}_tasks_tmp"
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)

    pipeline = KgPipe(
        tasks=pipeline_config.tasks,
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=tasks_tmp_dir,
        name=run_name,
    )

    pipeline.build(
        stable_files=True,
        configCatalog=pipeline_config.config_catalog,
        source=Data(path=source_path, format=DataFormat.RDF_NTRIPLES),
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES),
    )

    pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=True)
    return result_path


@pytest.mark.parametrize("config_idx", range(len(load_rdf_sampled_pipeline_configs())))
def test_rdf_pipeline_from_saved_sampled_configs(config_idx):
    """Runs KGpipe using PipelineConfigs materialized from the JSON fixture."""
    configs = load_rdf_sampled_pipeline_configs()
    assert configs, (
        "fixtures/rdf_sampled_pipeline_configs.json is missing or empty; "
        "run test_enumerate_all_valid_rdf_task_combinations_with_config_sampling"
    )

    pipeline_config = configs[config_idx]
    result_path = _run_rdf_pipeline_config(
        pipeline_config,
        tmp_dir=tmp_base_dir,
        run_name=f"saved_sample_config_idx_{config_idx}",
    )
    assert result_path.exists()


def test_sample_save_load_and_run_pipeline_config(tmp_path: Path):
    sampled_config = sample_valid_pipeline_config(
        RDF_SEARCH_SPACE,
        RDF_PIPELINE_LAYOUT,
        rng=random.Random(42),
    )
    original_key = pipeline_config_snapshot_key(sampled_config, RDF_SEARCH_SPACE)

    snapshot_path = tmp_path / "sampled_pipeline_config.json"
    save_pipeline_config_snapshot(snapshot_path, sampled_config)
    assert snapshot_path.exists()

    loaded_config = load_pipeline_config_snapshot(snapshot_path)
    loaded_key = pipeline_config_snapshot_key(loaded_config, RDF_SEARCH_SPACE)
    assert loaded_key == original_key

    result_path = _run_rdf_pipeline_config(
        loaded_config,
        tmp_dir=tmp_path,
        run_name="sample_save_load_run",
    )
    assert result_path.exists()


def test_sample_and_run_pipeline_config(tmp_path: Path):
    pipeline_config = sample_valid_pipeline_config(
        RDF_SEARCH_SPACE,
        RDF_PIPELINE_LAYOUT,
        rng=random.Random(43),
    )
    result_path = _run_rdf_pipeline_config(
        pipeline_config,
        tmp_dir=tmp_path,
        run_name="sample_and_run",
    )
    assert result_path.exists()
