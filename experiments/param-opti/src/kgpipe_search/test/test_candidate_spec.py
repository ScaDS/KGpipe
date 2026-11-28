"""Sampled search candidates are kgpipe.pipeline/v1 specifications.

The stored runs live under ``data/kgpipe-experiment-results/parameter_search/``.
Each candidate is a ``{task_keys, profiles}`` snapshot plus the plan that was
executed. The task tmp files are not in this checkout, so this test does not
call ``run()``. It rebuilds the candidate through the shared spec and checks
the task order, port refs, parameter values, fingerprint, and cache filename
against that stored plan. RDF and text sampled runs are both checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgpipe.common import Data, DataFormat, KgPipe
from kgpipe.generation.loaders import build_candidate, build_from_conf
from kgpipe.generation.spec import (
    PIPELINE_SPEC_SCHEMA,
    dump_pipeline_catalog,
    load_pipeline_catalog,
    normalize_parameter_key,
    pipeline_conf_from_plan,
    pipeline_task_name,
)
from kgpipe_search.configuration import pipeline_config_from_snapshot

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "data/kgpipe-experiment-results/parameter_search"


def _sampled_runs(directory: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for config_path in sorted(directory.glob("*.json")):
        if config_path.suffixes != [".json"]:
            continue
        plan_path = config_path.with_suffix(".plan.json")
        if plan_path.exists():
            pairs.append((config_path, plan_path))
    return pairs


def _endpoint(ref: str, steps) -> str | tuple[int, int]:
    if ref in {"$source", "$seed", "$result"}:
        return ref
    step_id, _port = ref.split(".", 1)
    for index, step in enumerate(steps):
        for output_index, output_ref in enumerate(step.outputs.values()):
            if step.id == step_id and output_ref == ref:
                return (index, output_index)
    raise AssertionError(f"Unresolved port ref {ref}")


def _assert_matches_stored_plan(candidate, plan, stored: dict) -> None:
    stored_steps = stored["steps"]
    assert [step["task"] for step in stored_steps] == [task.name for task in candidate.tasks]
    assert [step.task for step in plan.steps] == [task.name for task in candidate.tasks]

    source = stored["source"]["path"]
    seed = stored["seed"]["path"]
    result = stored["result"]["path"]
    produced: dict[str, tuple[int, int]] = {}

    for index, (stored_step, built_step, task) in enumerate(
        zip(stored_steps, plan.steps, candidate.tasks)
    ):
        input_ports = list(task.input_spec)
        output_ports = list(task.output_spec)
        assert len(stored_step["input"]) == len(input_ports)
        assert len(stored_step["output"]) == len(output_ports)
        for port, item in zip(input_ports, stored_step["input"]):
            path = item["path"]
            if path == source:
                expected = "$source"
            elif path == seed:
                expected = "$seed"
            elif path == result:
                expected = "$result"
            else:
                expected = produced[path]
            actual = _endpoint(built_step.inputs[port], plan.steps)
            assert actual == expected, (built_step.task, port, actual, expected)
        for output_index, (port, item) in enumerate(zip(output_ports, stored_step["output"])):
            path = item["path"]
            actual = _endpoint(built_step.outputs[port], plan.steps)
            if path == result:
                assert actual == "$result"
            else:
                produced[path] = (index, output_index)
                assert actual == (index, output_index)
                assert Path(item["path"]).name == Path(built_step.output[output_index].path).name


def _assert_parameter_values(candidate, plan) -> None:
    for task, step in zip(candidate.tasks, plan.steps):
        profile = candidate.config_catalog.get(task.name)
        if profile is None:
            assert step.parameters == {}
            continue
        for binding in profile.bindings:
            key = normalize_parameter_key(binding.parameter.name)
            assert step.parameters[key] == binding.value


@pytest.mark.parametrize("dirname", ["rdf_sampled", "text_sampled"])
def test_sampled_candidates_round_trip_through_spec(tmp_path: Path, dirname: str):
    directory = RESULTS / dirname
    runs = _sampled_runs(directory)
    assert runs, f"No sampled configs with plans in {directory}"
    assert len(runs) <= 15

    for config_path, plan_path in runs:
        snapshot = json.loads(config_path.read_text(encoding="utf-8"))
        stored = json.loads(plan_path.read_text(encoding="utf-8"))
        candidate = pipeline_config_from_snapshot(snapshot)
        work = tmp_path / config_path.stem

        source_format = DataFormat(stored["source"]["format"])
        seed_format = DataFormat(stored["seed"]["format"])
        result_format = DataFormat(stored["result"]["format"])
        seed = Data(path=work / "seed.nt", format=seed_format)
        source = Data(path=work / "source.nt", format=source_format)
        pipe, conf = build_candidate(
            config_path.stem,
            candidate.tasks,
            candidate.config_catalog,
            seed,
            str(work / "spec"),
            description=config_path.stem,
        )
        plan = pipe.build(
            stable_files=True,
            configCatalog=pipe.config_catalog,
            source=source,
            result=Data(path=work / "spec" / "result.nt", format=result_format),
        )

        legacy = KgPipe(
            tasks=candidate.tasks,
            seed=seed,
            data_dir=str(work / "legacy"),
            name=config_path.stem,
        )
        legacy_plan = legacy.build(
            stable_files=True,
            configCatalog=candidate.config_catalog,
            source=source,
            result=Data(path=work / "legacy" / "result.nt", format=result_format),
        )

        catalog_path = work / "pipeline.conf"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(dump_pipeline_catalog({config_path.stem: conf}))
        loaded = load_pipeline_catalog(catalog_path)
        assert loaded.schema_version == PIPELINE_SPEC_SCHEMA
        reloaded = loaded.root[config_path.stem]
        assert [pipeline_task_name(step) for step in reloaded.tasks] == [
            task.name for task in candidate.tasks
        ]
        reloaded_pipe = build_from_conf(
            config_path.stem,
            reloaded,
            seed,
            str(work / "yaml"),
            tasks=candidate.tasks,
        )
        reloaded_plan = reloaded_pipe.build(
            stable_files=True,
            configCatalog=reloaded_pipe.config_catalog,
            source=source,
            result=Data(path=work / "yaml" / "result.nt", format=result_format),
        )

        explicit = pipeline_conf_from_plan(plan, description=config_path.stem)
        explicit_pipe = build_from_conf(
            config_path.stem,
            explicit,
            seed,
            str(work / "explicit"),
            tasks=candidate.tasks,
        )
        explicit_plan = explicit_pipe.build(
            stable_files=True,
            configCatalog=explicit_pipe.config_catalog,
            source=source,
            result=Data(path=work / "explicit" / "result.nt", format=result_format),
        )

        assert conf.connect == "auto"
        assert explicit.connect == "explicit"
        _assert_parameter_values(candidate, plan)
        _assert_matches_stored_plan(candidate, plan, stored)
        assert plan.fingerprint
        assert plan.fingerprint == legacy_plan.fingerprint
        assert plan.fingerprint == reloaded_plan.fingerprint
        assert plan.fingerprint == explicit_plan.fingerprint
        assert [(step.inputs, step.outputs, step.parameters) for step in explicit_plan.steps] == [
            (step.inputs, step.outputs, step.parameters) for step in plan.steps
        ]
        assert plan.steps[0].output[0].path != legacy_plan.steps[0].output[0].path
