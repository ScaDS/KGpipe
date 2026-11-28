"""Published RDF and text snapshots keep their original hash, including exhaustive runs."""

from __future__ import annotations

import json
import random
from pathlib import Path

from kgpipe.generation.space import ConfigurationSpace, legacy_config_hash, legacy_snapshot_key, reject_task_combination
from kgpipe_search.configuration import (
    pipeline_config_from_snapshot,
    pipeline_config_snapshot_key,
    pipeline_config_to_snapshot,
    sample_valid_pipeline_config,
)
from kgpipe_search.definitions import (
    RDF_PIPELINE_LAYOUT,
    RDF_SEARCH_SPACE,
    TEXT_PIPELINE_LAYOUT,
    TEXT_SEARCH_SPACE,
    task_dict,
)

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "data/kgpipe-experiment-results/parameter_search"
DIRECTORIES = ("rdf_sampled", "text_sampled", "rdf", "text")


def _config_files(directory: Path) -> list[Path]:
    """Published config snapshots are ``{sha256}.json``. ``results.json`` is the run index."""
    files = []
    for path in sorted(directory.glob("*.json")):
        if path.suffixes != [".json"]:
            continue
        if len(path.stem) == 64 and all(char in "0123456789abcdef" for char in path.stem):
            files.append(path)
    return files


def test_published_snapshots_keep_their_hash_and_round_trip():
    for dirname in DIRECTORIES:
        directory = RESULTS / dirname
        files = _config_files(directory)
        assert files, f"No legacy snapshots in {directory}"
        for path in files:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            assert path.stem == legacy_config_hash(snapshot), path.name
            loaded = pipeline_config_from_snapshot(snapshot)
            rebuilt = pipeline_config_to_snapshot(snapshot["task_keys"], loaded)
            assert legacy_snapshot_key(rebuilt) == legacy_snapshot_key(snapshot)


def test_paper_search_space_reuses_task_constraints():
    rdf = ConfigurationSpace.from_legacy(RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT, task_dict)
    text = ConfigurationSpace.from_legacy(TEXT_SEARCH_SPACE, TEXT_PIPELINE_LAYOUT, task_dict)

    separate = [
        "paris_ontology_matching_task",
        "paris_entity_alignment_task",
        "aggregate_matching_results_task",
        "fusion_first_value_task",
    ]
    assert reject_task_combination(rdf, separate) is None
    assert reject_task_combination(rdf, ["paris_graph_alignment_task", "fusion_first_value_task"]) is None
    overlap = reject_task_combination(
        rdf,
        ["paris_ontology_matching_task", "paris_graph_alignment_task", "fusion_first_value_task"],
    )
    assert overlap is not None and "repeats categories" in overlap
    missing = reject_task_combination(
        rdf,
        ["paris_ontology_matching_task", "paris_entity_alignment_task", "fusion_first_value_task"],
    )
    assert missing is not None and "aggregate_matching_results" in missing
    assert reject_task_combination(text, ["corenlp_text_extraction_task"]) is not None

    first = sample_valid_pipeline_config(RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT, rng=random.Random(0))
    second = sample_valid_pipeline_config(RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT, rng=random.Random(0))
    assert pipeline_config_snapshot_key(first, RDF_SEARCH_SPACE) == pipeline_config_snapshot_key(
        second, RDF_SEARCH_SPACE
    )
