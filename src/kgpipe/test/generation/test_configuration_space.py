"""Configuration spaces: valid combinations, deterministic sampling, snapshot hashes."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType
from kgpipe.common.models import Data, DataFormat, KgTask
from kgpipe.generation.loaders import build_candidate, build_from_conf
from kgpipe.generation.space import (
    ConfigurationSpace,
    candidate_from_snapshot,
    candidate_snapshot,
    enumerate_task_combinations,
    iter_legacy_result_snapshots,
    legacy_config_hash,
    load_legacy_sample_file,
    read_legacy_snapshot,
    reject_task_combination,
    sample_candidate,
    sample_unique_snapshots,
)
from kgpipe.generation.spec import dump_pipeline_catalog, load_pipeline_catalog


def _spec(name: str, *, allowed: list, minimum: float | None = None) -> ConfigurationDefinition:
    return ConfigurationDefinition(
        name=name,
        parameters=[
            Parameter(
                name="threshold",
                datatype=ParameterType.number,
                default_value=0.5,
                required=True,
                allowed_values=allowed,
                minimum=minimum,
            )
        ],
    )


def _task(
    name: str,
    categories: list[str],
    *,
    allowed: list | None = None,
    minimum: float | None = None,
    inputs=None,
    outputs=None,
) -> KgTask:
    return KgTask(
        name=name,
        category=categories,
        input_spec=inputs or {"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
        output_spec=outputs or {"output": DataFormat.ER_JSON},
        function=lambda _inputs, _outputs: None,
        config_spec=_spec(name, allowed=allowed or [], minimum=minimum) if allowed is not None else None,
    )


def _space() -> ConfigurationSpace:
    onto = _task("onto", ["ontology"], allowed=[0.5, 0.9])
    entity = _task("entity", ["entity"], allowed=[0.5, 0.9])
    both = _task("both", ["ontology", "entity"], allowed=[0.5, 0.9])
    fuse = _task(
        "fuse",
        ["fusion"],
        inputs={
            "source": DataFormat.RDF_NTRIPLES,
            "kg": DataFormat.RDF_NTRIPLES,
            "matches": DataFormat.ER_JSON,
        },
        outputs={"output": DataFormat.RDF_NTRIPLES},
    )
    return ConfigurationSpace.from_tasks(
        [onto, entity, both, fuse],
        ["ontology", "entity", "fusion"],
    )


def test_valid_and_invalid_task_combinations():
    space = _space()
    combos = enumerate_task_combinations(space)
    assert combos == [
        ["onto", "entity", "fuse"],
        ["both", "fuse"],
    ]
    assert reject_task_combination(space, ["onto", "entity", "fuse"]) is None
    assert reject_task_combination(space, ["both", "fuse"]) is None

    overlap = reject_task_combination(space, ["onto", "both", "fuse"])
    assert overlap is not None
    assert "repeats categories" in overlap

    missing = reject_task_combination(space, ["onto", "entity"])
    assert missing is not None
    assert "fusion" in missing

    unknown = reject_task_combination(space, ["onto", "entity", "missing"])
    assert unknown is not None
    assert "missing" in unknown

    wrong_order = reject_task_combination(space, ["fuse", "onto", "entity"])
    assert wrong_order is not None
    assert "not a valid assignment" in wrong_order


def test_domain_must_satisfy_parameter_constraints():
    task = _task("onto", ["ontology"], allowed=[0.5, 0.9])
    with pytest.raises(ValueError, match="outside the allowed values"):
        ConfigurationSpace.from_tasks(
            [task],
            ["ontology"],
            domains={"onto": {"threshold": [0.2]}},
        )

    bounded = _task("onto", ["ontology"], allowed=[], minimum=0.5)
    with pytest.raises(ValueError, match="below the minimum"):
        ConfigurationSpace.from_tasks(
            [bounded],
            ["ontology"],
            domains={"onto": {"threshold": [0.4]}},
        )

    with pytest.raises(ValueError, match="not declared"):
        ConfigurationSpace.from_tasks(
            [task],
            ["ontology"],
            domains={"onto": {"unknown": [0.5]}},
        )


def test_sampling_is_deterministic_and_snapshot_hash_is_stable():
    space = _space()
    first = sample_candidate(space, random.Random(0))
    second = sample_candidate(space, random.Random(0))
    snapshot = candidate_snapshot(first)
    assert candidate_snapshot(second) == snapshot

    tasks_by_key = {entry.key: entry.task for entry in space.tasks}
    restored = candidate_from_snapshot(snapshot, tasks_by_key)
    assert candidate_snapshot(restored) == snapshot
    digest = legacy_config_hash(snapshot)
    assert legacy_config_hash(candidate_snapshot(restored)) == digest
    assert digest == legacy_config_hash(json.loads(json.dumps(snapshot)))

    again, _stats = sample_unique_snapshots(space, n=2, rng=random.Random(4))
    repeated, _stats = sample_unique_snapshots(space, n=2, rng=random.Random(4))
    assert again == repeated


def test_sampled_candidate_resolves_to_the_same_plan(tmp_path: Path):
    space = _space()
    tasks_by_key = {entry.key: entry.task for entry in space.tasks}
    candidate = sample_candidate(space, random.Random(2))
    snapshot = candidate_snapshot(candidate)
    restored = candidate_from_snapshot(snapshot, tasks_by_key)

    def _plan(item, dirname: str):
        seed = Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES)
        source = Data(path=tmp_path / "source.nt", format=DataFormat.RDF_NTRIPLES)
        pipe, conf = build_candidate(
            "candidate",
            item.tasks,
            item.profiles,
            seed,
            str(tmp_path / dirname),
        )
        plan = pipe.build(
            stable_files=True,
            configCatalog=pipe.config_catalog,
            source=source,
            result=Data(path=tmp_path / dirname / "result.nt", format=DataFormat.RDF_NTRIPLES),
        )
        return plan, conf

    plan, conf = _plan(restored, "spec")
    yaml_path = tmp_path / "pipeline.conf"
    yaml_path.write_text(dump_pipeline_catalog({"candidate": conf}))
    loaded = load_pipeline_catalog(yaml_path).root["candidate"]
    reloaded = build_from_conf(
        "candidate",
        loaded,
        Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES),
        str(tmp_path / "yaml"),
        tasks=restored.tasks,
    )
    reloaded_plan = reloaded.build(
        stable_files=True,
        configCatalog=reloaded.config_catalog,
        source=Data(path=tmp_path / "source.nt", format=DataFormat.RDF_NTRIPLES),
        result=Data(path=tmp_path / "yaml" / "result.nt", format=DataFormat.RDF_NTRIPLES),
    )
    assert plan.fingerprint
    assert plan.fingerprint == reloaded_plan.fingerprint
    assert [step.task for step in plan.steps] == [task.name for task in restored.tasks]


def test_legacy_sample_file_and_result_hash(tmp_path: Path):
    snapshot = {
        "profiles": {
            "onto": {
                "bindings": [{"parameter": "threshold", "value": 0.5}],
                "profile_name": "onto_threshold=0.5",
            }
        },
        "task_keys": ["onto_task"],
    }
    digest = legacy_config_hash(snapshot)
    result = tmp_path / f"{digest}.json"
    result.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    assert read_legacy_snapshot(result)["task_keys"] == ["onto_task"]
    listed = list(iter_legacy_result_snapshots(tmp_path))
    assert [path.name for path, _item in listed] == [result.name]

    renamed = tmp_path / "not-the-hash.json"
    renamed.write_text(result.read_text())
    with pytest.raises(ValueError, match="does not match the legacy config hash"):
        read_legacy_snapshot(renamed)

    fixture = tmp_path / "samples.json"
    fixture.write_text(json.dumps({"version": 1, "samples": [snapshot]}) + "\n")
    assert load_legacy_sample_file(fixture) == [snapshot]
    with pytest.raises(ValueError, match="Unsupported"):
        load_legacy_sample_file(fixture, expected_version=2)


def test_categories_come_from_the_task_when_the_space_does_not_override_them():
    task = _task("onto", ["ontology"], allowed=[0.5, 0.9])
    space = ConfigurationSpace.from_tasks([task], ["ontology"])
    assert space.by_key["onto"].categories == ["ontology"]
    assert space.by_key["onto"].domain["threshold"] == [0.5, 0.9]
    assert reject_task_combination(space, ["onto"]) is None
