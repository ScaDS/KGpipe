from pathlib import Path

import pytest

from kgpipe import Pipeline
from kgpipe.api import CatalogValidationError, PipelineConf, PipelineTaskConf
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType
from kgpipe.common.models import Data, DataFormat, KgPipe, KgTask
from kgpipe.common.registry import Registry
from kgpipe.generation.spec import PIPELINE_SPEC_SCHEMA


def _write(_inputs, outputs):
    for data in outputs.values():
        data.path.parent.mkdir(parents=True, exist_ok=True)
        data.path.write_text("ok\n")


def _boom(_inputs, _outputs):
    raise RuntimeError("nope")


def _noop(_inputs, _outputs):
    return None


THRESHOLD = ConfigurationDefinition(
    name="api-threshold",
    parameters=[
        Parameter(
            name="entity matching threshold",
            datatype=ParameterType.number,
            default_value=0.9,
        ),
    ],
)


def _task(name: str, input_spec, output_spec, function=_noop, **kwargs) -> KgTask:
    task = KgTask(
        name=name,
        input_spec=input_spec,
        output_spec=output_spec,
        function=function,
        **kwargs,
    )
    return Registry.add_task(task.name, task)


def _register() -> None:
    _task(
        "api_match",
        {"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
        {"output": DataFormat.PARIS_CSV},
    )
    _task(
        "api_exchange",
        {"input": DataFormat.PARIS_CSV},
        {"output": DataFormat.ER_JSON},
    )
    _task(
        "api_fuse",
        {
            "source": DataFormat.RDF_NTRIPLES,
            "kg": DataFormat.RDF_NTRIPLES,
            "matches1": DataFormat.ER_JSON,
        },
        {"output": DataFormat.RDF_NTRIPLES},
        config_spec=THRESHOLD,
    )
    _task(
        "api_types",
        {"source": DataFormat.RDF_NTRIPLES},
        {"result": DataFormat.RDF_NTRIPLES},
        function=_write,
    )
    _task(
        "api_copy",
        {"source": DataFormat.RDF_NTRIPLES},
        {"result": DataFormat.RDF_NTRIPLES},
        function=_write,
    )
    _task(
        "api_boom",
        {"source": DataFormat.RDF_NTRIPLES},
        {"result": DataFormat.RDF_NTRIPLES},
        function=_boom,
    )
    _task(
        "api_docker",
        {"source": DataFormat.RDF_NTRIPLES},
        {"result": DataFormat.RDF_NTRIPLES},
        backend="docker",
    )


_register()


def _chain_tasks():
    return [
        PipelineTaskConf(id="match", task="api_match", backend="python"),
        "api_exchange",
        "api_fuse",
        "api_types",
    ]


def _pipeline(tmp_path: Path, **overrides) -> Pipeline:
    source = tmp_path / "source.nt"
    seed = tmp_path / "seed.nt"
    source.write_text("<s> <p> <o> .\n")
    seed.write_text("<s> <p> <o> .\n")
    kwargs = dict(
        name="rdf",
        description="Align source RDF with a seed KG",
        tasks=_chain_tasks(),
        parameters={"entity_matching_threshold": 0.99},
        connect="auto",
        backend="python",
        inputs={"source": "nt", "seed": "nt"},
        output={"result": "nt"},
        source=source,
        seed=seed,
        result=tmp_path / "out" / "result.nt",
        data_dir=tmp_path / "out",
    )
    kwargs.update(overrides)
    return Pipeline(**kwargs)


def _yaml(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "rdf:",
            '  description: "Align source RDF with a seed KG"',
            "  connect: auto",
            "  backend: python",
            "  inputs:",
            "    source: {role: source, format: nt}",
            "    seed: {role: seed, format: nt}",
            "  output:",
            "    result: {role: result, format: nt}",
            "  parameters:",
            "    entity_matching_threshold: 0.99",
            "  tasks:",
            "    - id: match",
            "      task: api_match",
            "      backend: python",
            "    - api_exchange",
            "    - api_fuse",
            "    - api_types",
            "",
        ])
    )
    return path


def test_python_and_yaml_plans_match(tmp_path: Path):
    python_pipeline = _pipeline(tmp_path)
    yaml_pipeline = Pipeline.from_yaml(
        _yaml(tmp_path),
        source=tmp_path / "source.nt",
        seed=tmp_path / "seed.nt",
        result=tmp_path / "yaml" / "result.nt",
        data_dir=tmp_path / "yaml",
        environ={},
    )
    python_plan = python_pipeline.plan()
    yaml_plan = yaml_pipeline.plan()
    assert python_plan.fingerprint == yaml_plan.fingerprint
    assert [(step.inputs, step.outputs, step.parameters) for step in python_plan.steps] == [
        (step.inputs, step.outputs, step.parameters) for step in yaml_plan.steps
    ]
    assert python_plan.steps[2].parameters["entity_matching_threshold"] == 0.99
    assert python_pipeline.to_kgpipe().step_specs[0].backend == "python"
    assert yaml_pipeline.conf.parameters["entity_matching_threshold"] == 0.99


def test_from_conf_matches_pipeline_constructor(tmp_path: Path):
    direct = _pipeline(tmp_path)
    conf = PipelineConf(
        description="Align source RDF with a seed KG",
        connect="auto",
        backend="python",
        inputs={"source": {"role": "source", "format": "nt"}, "seed": {"role": "seed", "format": "nt"}},
        output={"result": {"role": "result", "format": "nt"}},
        parameters={"entity_matching_threshold": 0.99},
        tasks=_chain_tasks(),
    )
    wrapped = Pipeline.from_conf(
        "rdf",
        conf,
        source=tmp_path / "source.nt",
        seed=tmp_path / "seed.nt",
        result=tmp_path / "conf" / "result.nt",
        data_dir=tmp_path / "conf",
    )
    assert direct.plan().fingerprint == wrapped.plan().fingerprint
    assert wrapped.conf is conf


def test_declared_default_is_on_the_plan(tmp_path: Path):
    fuse = _pipeline(tmp_path, parameters={})
    assert fuse.plan().steps[2].parameters["entity_matching_threshold"] == 0.9


def test_invalid_parameter_and_unknown_task_name_the_pipeline(tmp_path: Path):
    with pytest.raises(CatalogValidationError, match="Pipeline 'rdf'"):
        _pipeline(tmp_path, parameters={"entity_matching_threshold": "nope"}).validate()
    with pytest.raises(CatalogValidationError, match="not_a_parameter"):
        _pipeline(tmp_path, parameters={"not_a_parameter": 1}).plan()
    with pytest.raises(CatalogValidationError, match="missing_api_task"):
        Pipeline(tasks=["missing_api_task"], name="demo").run()
    with pytest.raises(CatalogValidationError, match="backend 'docker'"):
        _pipeline(tmp_path, tasks=["api_match"], backend="docker").validate()


def test_incompatible_tasks_fail_before_execution(tmp_path: Path):
    called = {"ran": False}

    def _mark(_inputs, _outputs):
        called["ran"] = True

    task = _task(
        "api_json_only",
        {"input": DataFormat.JSON},
        {"output": DataFormat.JSON},
        function=_mark,
    )
    pipeline = Pipeline(
        tasks=[task, "api_types"],
        source=tmp_path / "source.json",
        seed=tmp_path / "seed.json",
        result=tmp_path / "result.nt",
        data_dir=tmp_path / "work",
    )
    with pytest.raises(CatalogValidationError, match="Cannot connect task"):
        pipeline.validate()
    assert called["ran"] is False


def test_run_returns_structured_result(tmp_path: Path):
    source = tmp_path / "source.nt"
    source.write_text("<s> <p> <o> .\n")
    pipeline = Pipeline(
        tasks=["api_copy"],
        name="copy",
        source=source,
        seed=tmp_path / "seed.nt",
        result=tmp_path / "result.nt",
        data_dir=tmp_path / "work",
    )
    outcome = pipeline.run()
    assert outcome.status == "success"
    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.result.path.read_text() == "ok\n"
    assert [report.status for report in outcome.reports] == ["success"]
    assert outcome.plan.fingerprint == pipeline.plan().fingerprint
    dumped = outcome.model_dump(mode="json")
    assert dumped["status"] == "success"
    assert dumped["plan"]["fingerprint"] == outcome.plan.fingerprint
    assert dumped["reports"][0]["task_name"] == "api_copy"

    again = pipeline.run()
    assert again.status == "skipped"
    assert again.reports[0].status == "skipped"


def test_failed_task_and_missing_input_stay_on_the_result(tmp_path: Path):
    source = tmp_path / "source.nt"
    source.write_text("<s> <p> <o> .\n")
    failed = Pipeline(
        tasks=["api_boom"],
        name="boom",
        source=source,
        seed=tmp_path / "seed.nt",
        result=tmp_path / "result.nt",
        data_dir=tmp_path / "work",
    ).run()
    assert failed.status == "failed"
    assert failed.error == "nope"
    assert failed.reports[0].status == "failed"

    missing = Pipeline(
        tasks=["api_copy"],
        name="missing",
        source=tmp_path / "absent.nt",
        seed=tmp_path / "seed.nt",
        result=tmp_path / "result.nt",
        data_dir=tmp_path / "missing-work",
    ).run()
    assert missing.status == "failed"
    assert missing.error is not None
    assert "absent.nt" in missing.error
    assert missing.reports == []


def test_live_task_does_not_need_the_registry(tmp_path: Path):
    task = KgTask(
        name="api_unregistered",
        input_spec={"source": DataFormat.RDF_NTRIPLES},
        output_spec={"result": DataFormat.RDF_NTRIPLES},
        function=_write,
    )
    source = tmp_path / "source.nt"
    source.write_text("ok\n")
    outcome = Pipeline(
        tasks=[task],
        source=source,
        seed=tmp_path / "seed.nt",
        result=tmp_path / "result.nt",
        data_dir=tmp_path / "work",
    ).run()
    assert outcome.status == "success"
    assert outcome.plan.steps[0].task == "api_unregistered"


def test_from_kgpipe_preserves_the_built_plan(tmp_path: Path):
    source = Data(path=tmp_path / "source.nt", format=DataFormat.RDF_NTRIPLES)
    seed = Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES)
    legacy_result = Data(path=tmp_path / "legacy" / "result.nt", format=DataFormat.RDF_NTRIPLES)
    tasks = [Registry.get_task(name) for name in ("api_match", "api_exchange", "api_fuse", "api_types")]
    legacy = KgPipe(tasks=tasks, seed=seed, data_dir=str(tmp_path / "legacy"), name="rdf")
    legacy.build(source=source, result=legacy_result, stable_files=True)

    wrapped = Pipeline.from_kgpipe(legacy)
    assert wrapped.plan().fingerprint == legacy.plan.fingerprint
    with pytest.raises(CatalogValidationError, match="already has a plan"):
        Pipeline.from_kgpipe(legacy, source=source)

    modern = Pipeline(
        tasks=tasks,
        name="rdf",
        source=source,
        seed=seed,
        result=Data(path=tmp_path / "modern" / "result.nt", format=DataFormat.RDF_NTRIPLES),
        data_dir=tmp_path / "modern",
    )
    # Hand-built KgPipe does not bind config_spec defaults. The public API does,
    # which is the same plan a YAML catalog produces.
    assert modern.plan().steps[2].parameters["entity_matching_threshold"] == 0.9
    assert legacy.plan.steps[2].parameters == {}
    assert wrapped.plan().fingerprint == legacy.plan.fingerprint


def test_unbuilt_kgpipe_builds_through_the_adapter(tmp_path: Path):
    source = Data(path=tmp_path / "source.nt", format=DataFormat.RDF_NTRIPLES)
    seed = Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES)
    task = Registry.get_task("api_copy")
    legacy = KgPipe(tasks=[task], seed=seed, data_dir=str(tmp_path / "legacy"), name="copy")
    with pytest.raises(CatalogValidationError, match="has no plan"):
        Pipeline.from_kgpipe(legacy)
    wrapped = Pipeline.from_kgpipe(
        legacy,
        source=source,
        result=tmp_path / "legacy" / "result.nt",
    )
    modern = Pipeline(
        tasks=[task],
        name="copy",
        source=source,
        seed=seed,
        result=tmp_path / "modern" / "result.nt",
        data_dir=tmp_path / "modern",
    )
    assert wrapped.plan().fingerprint == modern.plan().fingerprint
    assert legacy.plan.fingerprint == wrapped.plan().fingerprint


def test_explicit_python_and_yaml_plans_match(tmp_path: Path):
    tasks = [
        {
            "id": "match",
            "task": "api_match",
            "inputs": {"source": "$source", "kg": "$seed"},
            "outputs": {"output": "match.output"},
        },
        {
            "id": "exchange",
            "task": "api_exchange",
            "inputs": {"input": "match.output"},
            "outputs": {"output": "exchange.output"},
        },
    ]
    python_pipeline = Pipeline(
        name="rdf",
        tasks=tasks,
        connect="explicit",
        source=tmp_path / "source.nt",
        seed=tmp_path / "seed.nt",
        result=tmp_path / "python" / "result.nt",
        data_dir=tmp_path / "python",
    )
    yaml_path = tmp_path / "explicit.yaml"
    yaml_path.write_text(
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "rdf:",
            '  description: "explicit"',
            "  connect: explicit",
            "  tasks:",
            "    - id: match",
            "      task: api_match",
            "      inputs: {source: $source, kg: $seed}",
            "      outputs: {output: match.output}",
            "    - id: exchange",
            "      task: api_exchange",
            "      inputs: {input: match.output}",
            "      outputs: {output: exchange.output}",
            "",
        ])
    )
    yaml_pipeline = Pipeline.from_yaml(
        yaml_path,
        source=tmp_path / "source.nt",
        seed=tmp_path / "seed.nt",
        result=tmp_path / "yaml" / "result.nt",
        data_dir=tmp_path / "yaml",
        environ={},
    )
    assert python_pipeline.plan().fingerprint == yaml_pipeline.plan().fingerprint
    assert python_pipeline.plan().steps[0].inputs == {"source": "$source", "kg": "$seed"}
    assert python_pipeline.plan().steps[1].inputs == {"input": "match.output"}


def test_seed_and_target_must_be_the_same_file(tmp_path: Path):
    with pytest.raises(CatalogValidationError, match="seed and target"):
        Pipeline(
            tasks=["api_copy"],
            source=tmp_path / "source.nt",
            seed=tmp_path / "seed.nt",
            target=tmp_path / "other.nt",
            result=tmp_path / "result.nt",
            data_dir=tmp_path,
        ).plan()
