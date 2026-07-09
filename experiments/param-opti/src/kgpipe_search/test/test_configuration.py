from kgpipe_search.definitions import PipelineLayout, RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT, TEXT_SEARCH_SPACE, TEXT_PIPELINE_LAYOUT
from kgpipe_search.configuration import (
    sample_valid_pipeline_config, 
    enumerate_valid_task_combinations, sample_config_catalog_for_task_combo, enumerate_exhaustive_pipeline_config_snapshots, pipeline_config_to_snapshot,
    print_pipeline_config_short,
    sample_unique_pipeline_config_snapshots_per_combo,
)
from kgpipe_search.definitions import RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE, _RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION
from kgpipe_search.definitions import RDF_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE, _RDF_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION
from kgpipe_search.definitions import RDF_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE, _RDF_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION
from kgpipe_search.definitions import TEXT_SAMPLED_PIPELINE_CONFIGS_FIXTURE, _TEXT_PIPELINE_CONFIG_SNAPSHOT_VERSION
from kgpipe_search.definitions import TEXT_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE, _TEXT_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION
from kgpipe_search.definitions import TEXT_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE, _TEXT_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION
import json

def test_sample_valid_rdf_pipeline_config():
    pipeline_layout = PipelineLayout(
        allowed_task_categories=["ontology_matching", "entity_matching", "aggregate_matching_results", "fusion"]
    )
    pipeline_config = sample_valid_pipeline_config(RDF_SEARCH_SPACE, pipeline_layout)
    print_pipeline_config_short(pipeline_config)

def test_enumerate_all_valid_rdf_task_combinations_no_config_sampling():
    print("enumerate_all_valid_rdf_task_combinations_no_config_sampling")
    combos = enumerate_valid_task_combinations(RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT)

    for combo in combos:
        print(combo)

    # With current SEARCH_SPACE:
    # - ontology_matching can be satisfied by paris_ontology_matching_task, paris_entity_alignment_task, paris_graph_alignment_task
    # - entity_matching can be satisfied by paris_entity_alignment_task, paris_graph_alignment_task (and may be skipped if already covered)
    # - fusion must be satisfied by fusion_first_value_task
    # expected = {
    #     ("paris_ontology_matching_task", "paris_entity_alignment_task", "fusion_first_value_task"),
    #     ("paris_ontology_matching_task", "paris_graph_alignment_task", "fusion_first_value_task"),
    #     ("paris_graph_alignment_task", "fusion_first_value_task"),
    # }

   # assert set(tuple(c) for c in combos) == expected


import random
from typing import List, Dict, Any


def _print_unique_sampling_stats(stats: Dict[str, Any]) -> None:
    print()
    print("unique sampling statistics")
    print(f"requested n per combo: {stats['requested_n']}")
    print(f"total combos: {stats['total_combos']}")
    print(f"total snapshots: {stats['total_snapshots']}")
    print(f"combos exhausted before n: {stats['combos_exhausted']}")
    for row in stats["combos"]:
        status = "EXHAUSTED" if row["exhausted"] else "ok"
        print(
            f"  {row['task_keys']}: sampled {row['sampled']}/{row['requested']} "
            f"(available {row['available_profiles']}) [{status}]"
        )


def test_enumerate_all_valid_rdf_task_combinations_with_config_sampling():
    print("enumerate_all_valid_rdf_task_combinations_with_config_sampling")
    n = 1
    rng = random.Random(0)

    combos = enumerate_valid_task_combinations(RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT)

    total_config_count = 0
    snapshots: List[Dict[str, Any]] = []

    for combo in combos:
        print()
        print("combo:", combo)
        for i in range(n):
            total_config_count += 1
            print(f"sample {total_config_count}/{len(combos) * n}")
            pipeline_config = sample_config_catalog_for_task_combo(
                RDF_SEARCH_SPACE, combo, rng=rng
            )

            print_pipeline_config_short(pipeline_config)
            snapshots.append(pipeline_config_to_snapshot(combo, pipeline_config))

    RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE.write_text(
        json.dumps(
            {"version": _RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION, "samples": snapshots},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_enumerate_all_valid_rdf_task_combinations_with_unique_config_sampling():
    print("enumerate_all_valid_rdf_task_combinations_with_unique_config_sampling")
    n = 10
    rng = random.Random(0)

    snapshots, stats = sample_unique_pipeline_config_snapshots_per_combo(
        RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT, n=n, rng=rng
    )
    _print_unique_sampling_stats(stats)

    for combo_row in stats["combos"]:
        combo_task_keys = combo_row["task_keys"]
        combo_snapshots = [s for s in snapshots if s["task_keys"] == combo_task_keys]
        serialized = [json.dumps(s, sort_keys=True) for s in combo_snapshots]
        assert len(set(serialized)) == len(serialized)
        assert len(serialized) == combo_row["sampled"]

    RDF_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    RDF_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE.write_text(
        json.dumps(
            {"version": _RDF_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION, "samples": snapshots},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_sample_valid_text_pipeline_config():
    pipeline_config = sample_valid_pipeline_config(TEXT_SEARCH_SPACE, TEXT_PIPELINE_LAYOUT)
    print_pipeline_config_short(pipeline_config)


def test_enumerate_all_valid_text_task_combinations_no_config_sampling():
    print("enumerate_all_valid_text_task_combinations_no_config_sampling")
    combos = enumerate_valid_task_combinations(TEXT_SEARCH_SPACE, TEXT_PIPELINE_LAYOUT)
    for combo in combos:
        print(combo)

def test_enumerate_all_valid_text_task_combinations_with_config_sampling():
    print("enumerate_all_valid_text_task_combinations_with_config_sampling")
    n = 1
    rng = random.Random(0)

    combos = enumerate_valid_task_combinations(TEXT_SEARCH_SPACE, TEXT_PIPELINE_LAYOUT)

    total_config_count = 0
    snapshots: List[Dict[str, Any]] = []

    for combo in combos:
        print()
        print("combo:", combo)
        for i in range(n):
            total_config_count += 1
            print(f"sample {total_config_count}/{len(combos) * n}")
            pipeline_config = sample_config_catalog_for_task_combo(
                TEXT_SEARCH_SPACE, combo, rng=rng
            )
            print_pipeline_config_short(pipeline_config)
            snapshots.append(pipeline_config_to_snapshot(combo, pipeline_config))

    TEXT_SAMPLED_PIPELINE_CONFIGS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    TEXT_SAMPLED_PIPELINE_CONFIGS_FIXTURE.write_text(
        json.dumps(
            {"version": _TEXT_PIPELINE_CONFIG_SNAPSHOT_VERSION, "samples": snapshots},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_enumerate_all_valid_text_task_combinations_with_unique_config_sampling():
    print("enumerate_all_valid_text_task_combinations_with_unique_config_sampling")
    n = 3
    rng = random.Random(0)

    snapshots, stats = sample_unique_pipeline_config_snapshots_per_combo(
        TEXT_SEARCH_SPACE, TEXT_PIPELINE_LAYOUT, n=n, rng=rng
    )
    _print_unique_sampling_stats(stats)

    for combo_row in stats["combos"]:
        combo_task_keys = combo_row["task_keys"]
        combo_snapshots = [s for s in snapshots if s["task_keys"] == combo_task_keys]
        serialized = [json.dumps(s, sort_keys=True) for s in combo_snapshots]
        assert len(set(serialized)) == len(serialized)
        assert len(serialized) == combo_row["sampled"]

    TEXT_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    TEXT_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE.write_text(
        json.dumps(
            {"version": _TEXT_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION, "samples": snapshots},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_enumerate_all_valid_text_task_combinations_with_config_sampling_exhaustive():
    print("enumerate_all_valid_text_task_combinations_with_config_sampling_exhaustive")
    all_snapshots = enumerate_exhaustive_pipeline_config_snapshots(
        TEXT_SEARCH_SPACE, TEXT_PIPELINE_LAYOUT
    )
    serialized = [json.dumps(s, sort_keys=True) for s in all_snapshots]
    assert len(set(serialized)) == len(serialized)

    TEXT_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    TEXT_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE.write_text(
        json.dumps(
            {"version": _TEXT_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION, "samples": all_snapshots},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_enumerate_all_valid_rdf_task_combinations_with_config_sampling_exhaustive():
    print("enumerate_all_valid_rdf_task_combinations_with_config_sampling_exhaustive")
    all_snapshots = enumerate_exhaustive_pipeline_config_snapshots(
        RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT
    )
    serialized = [json.dumps(s, sort_keys=True) for s in all_snapshots]
    assert len(set(serialized)) == len(serialized)

    RDF_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    RDF_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE.write_text(
        json.dumps(
            {"version": _RDF_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION, "samples": all_snapshots},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )



# def test_rdf_pipeline_from_config():
#     pipeline_config = sample_valid_pipeline_config(RDF_SEARCH_SPACE, PipelineLayout(allowed_task_categories=["entity_matching", "fusion"]))

#     seed_path = tmp_base_dir / "seed.nt"
#     source_path = tmp_base_dir / "source.nt"
#     result_path = tmp_base_dir / "result.nt"
#     tasks_tmp_dir = tmp_base_dir / "tasks_tmp"
#     tasks_tmp_dir.mkdir(parents=True, exist_ok=True)

#     # Ensure inputs exist for pipeline execution.
#     seed_path.write_text("<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n")
#     source_path.write_text("<http://example.org/s2> <http://example.org/p> <http://example.org/o> .\n")

#     pipeline = KgPipe(
#         tasks=pipeline_config.tasks, 
#         seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
#         data_dir=tasks_tmp_dir,
#         name="test_pipeline")

#     pipeline.build(
#         stable_files=True,
#         configCatalog=pipeline_config.config_catalog,
#         source=Data(path=source_path, format=DataFormat.RDF_NTRIPLES), 
#         result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES))

#     pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=True)