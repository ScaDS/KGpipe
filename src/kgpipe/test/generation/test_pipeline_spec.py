import json
from pathlib import Path

import pytest

from kgpipe.common.model.configuration import (
    ConfigurationDefinition,
    ConfigurationProfile,
    Parameter,
    ParameterBinding,
    ParameterType,
)
from kgpipe.common.models import Data, DataFormat, KgTask
from kgpipe.common.registry import Registry
from kgpipe.generation.loaders import build_candidate, build_from_conf
from kgpipe.generation.spec import (
    PIPELINE_SPEC_SCHEMA,
    PipelineConf,
    PipelineTaskConf,
    dump_pipeline_catalog,
    load_pipeline_catalog,
    pipeline_conf_from_plan,
)


def _register(task: KgTask) -> KgTask:
    Registry.add_task(task.name, task)
    return task


def _rdf_tasks() -> None:
    _register(KgTask(
        name="match",
        input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"output": DataFormat.PARIS_CSV},
        function=lambda _inputs, _outputs: None,
    ))
    _register(KgTask(
        name="exchange",
        input_spec={"input": DataFormat.PARIS_CSV},
        output_spec={"output": DataFormat.ER_JSON},
        function=lambda _inputs, _outputs: None,
    ))
    _register(KgTask(
        name="fuse",
        input_spec={
            "source": DataFormat.RDF_NTRIPLES,
            "kg": DataFormat.RDF_NTRIPLES,
            "matches1": DataFormat.ER_JSON,
        },
        output_spec={"output": DataFormat.RDF_NTRIPLES},
        function=lambda _inputs, _outputs: None,
        config_spec=ConfigurationDefinition(
            name="fuse-config",
            parameters=[
                Parameter(
                    name="entity matching threshold",
                    datatype=ParameterType.number,
                    default_value=0.9,
                ),
                Parameter(
                    name="relation matching threshold",
                    datatype=ParameterType.number,
                    default_value=0.5,
                ),
            ],
        ),
    ))
    _register(KgTask(
        name="types",
        input_spec={"source": DataFormat.RDF_NTRIPLES},
        output_spec={"result": DataFormat.RDF_NTRIPLES},
        function=lambda _inputs, _outputs: None,
    ))


def _build_rdf(tmp_path: Path, conf: PipelineConf, dirname: str):
    seed = Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES)
    source = Data(path=tmp_path / "source.nt", format=DataFormat.RDF_NTRIPLES)
    result = Data(path=tmp_path / dirname / "result.nt", format=DataFormat.RDF_NTRIPLES)
    pipe = build_from_conf("rdf", conf, seed, str(tmp_path / dirname))
    plan = pipe.build(source=source, result=result, stable_files=True)
    return plan


def test_legacy_catalogs_load():
    repo = Path(__file__).resolve().parents[4]
    moviekg = load_pipeline_catalog(repo / "experiments/moviekg/pipeline.conf")
    dummy = load_pipeline_catalog(repo / "src/kgpipe/test/test_data/dummy_pipeline.conf")

    assert moviekg.schema_version == PIPELINE_SPEC_SCHEMA
    rdf = moviekg.root["rdf_base"]
    assert rdf.connect == "auto"
    assert rdf.config["ENTITY_MATCHING_THRESHOLD"] == "0.99"
    assert rdf.tasks[:2] == ["paris_entity_matching", "paris_exchange"]
    assert dummy.root["dummy"].tasks == ["dummy_task"]


def test_catalog_schema_key_is_not_a_pipeline(tmp_path: Path):
    path = tmp_path / "pipeline.conf"
    path.write_text(
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "versioned"',
            "  tasks:",
            "    - demo_task",
            "",
        ])
    )
    catalog = load_pipeline_catalog(path)
    assert list(catalog.root) == ["demo"]
    assert catalog.schema_version == PIPELINE_SPEC_SCHEMA


def test_unsupported_schema_is_rejected(tmp_path: Path):
    path = tmp_path / "pipeline.conf"
    path.write_text("schema: kgpipe.pipeline/v0\ndemo:\n  description: d\n  tasks: []\n")
    with pytest.raises(ValueError, match="Unsupported pipeline catalog schema"):
        load_pipeline_catalog(path)


def test_auto_connect_records_ports_and_stable_fingerprint(tmp_path: Path):
    _rdf_tasks()
    conf = PipelineConf(
        description="auto",
        config={"ENTITY_MATCHING_THRESHOLD": "0.99", "UNRELATED_ENV": "1"},
        tasks=["match", "exchange", "fuse", "types"],
    )
    plan_a = _build_rdf(tmp_path, conf, "a")
    plan_b = _build_rdf(tmp_path, conf, "b")

    assert [(step.id, step.inputs, step.outputs) for step in plan_a.steps] == [
        ("match", {"source": "$source", "kg": "$seed"}, {"output": "match.output"}),
        ("exchange", {"input": "match.output"}, {"output": "exchange.output"}),
        (
            "fuse",
            {"source": "$source", "kg": "$seed", "matches1": "exchange.output"},
            {"output": "fuse.output"},
        ),
        ("types", {"source": "fuse.output"}, {"result": "$result"}),
    ]
    assert plan_a.steps[2].parameters == {
        "entity_matching_threshold": 0.99,
        "relation_matching_threshold": 0.5,
    }
    assert plan_a.fingerprint == plan_b.fingerprint
    assert plan_a.steps[0].output[0].path != plan_b.steps[0].output[0].path
    assert plan_a.connect == "auto"

    explicit = PipelineConf(
        description="explicit",
        connect="explicit",
        config={"ENTITY_MATCHING_THRESHOLD": "0.99"},
        tasks=[
            PipelineTaskConf(
                id=step.id,
                name=step.task,
                inputs=dict(step.inputs),
                outputs=dict(step.outputs),
                parameters=dict(step.parameters),
            )
            for step in plan_a.steps
        ],
    )
    plan_explicit = _build_rdf(tmp_path, explicit, "explicit")
    assert plan_explicit.fingerprint == plan_a.fingerprint
    assert [(step.inputs, step.outputs) for step in plan_explicit.steps] == [
        (step.inputs, step.outputs) for step in plan_a.steps
    ]


def test_step_parameters_override_pipeline_and_legacy_values(tmp_path: Path):
    _rdf_tasks()
    conf = PipelineConf(
        description="overrides",
        config={
            "ENTITY_MATCHING_THRESHOLD": "0.99",
            "RELATION_MATCHING_THRESHOLD": "0.2",
        },
        parameters={"relation_matching_threshold": 0.1},
        tasks=[
            "match",
            "exchange",
            PipelineTaskConf(name="fuse", parameters={"entity_matching_threshold": 0.8}),
            "types",
        ],
    )
    plan = _build_rdf(tmp_path, conf, "overrides")
    assert plan.steps[2].parameters == {
        "entity_matching_threshold": 0.8,
        "relation_matching_threshold": 0.1,
    }


def test_unknown_pipeline_parameter_is_rejected(tmp_path: Path):
    _rdf_tasks()
    conf = PipelineConf(
        description="bad parameter",
        parameters={"not_a_parameter": 1},
        tasks=["match", "exchange", "fuse", "types"],
    )
    seed = Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES)
    with pytest.raises(ValueError, match="not_a_parameter"):
        build_from_conf("rdf", conf, seed, str(tmp_path))


def test_explicit_bindings_require_connect_explicit():
    conf = PipelineConf(
        description="implicit bindings",
        tasks=[PipelineTaskConf(name="match", inputs={"source": "$source"})],
    )
    with pytest.raises(ValueError, match="connect: explicit"):
        build_from_conf(
            "rdf",
            conf,
            Data(path="seed.nt", format=DataFormat.RDF_NTRIPLES),
            "/tmp",
        )


def test_explicit_format_mismatch_and_unknown_ref(tmp_path: Path):
    _rdf_tasks()
    seed = Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES)
    source = Data(path=tmp_path / "source.json", format=DataFormat.JSON)
    result = Data(path=tmp_path / "result.nt", format=DataFormat.RDF_NTRIPLES)
    conf = PipelineConf(
        description="bad ref",
        connect="explicit",
        tasks=[
            PipelineTaskConf(
                id="match",
                name="match",
                inputs={"source": "$missing", "kg": "$seed"},
                outputs={"output": "match.output"},
            )
        ],
    )
    pipe = build_from_conf("rdf", conf, seed, str(tmp_path))
    with pytest.raises(ValueError, match="unknown '\\$missing'"):
        pipe.build(source=source, result=result)

    conf.tasks = [
        PipelineTaskConf(
            id="match",
            name="match",
            inputs={"source": "$source", "kg": "$seed"},
            outputs={"output": "match.output"},
        )
    ]
    pipe = build_from_conf("rdf", conf, seed, str(tmp_path))
    with pytest.raises(ValueError, match="expects nt"):
        pipe.build(source=source, result=result)


def test_fusion_thresholds_bind_from_pipeline_config():
    import kgpipe_tasks.entity_resolution.fusion.preference  # noqa: F401
    from kgpipe.common.registry import Registry
    from kgpipe.generation.spec import bind_config_catalog, normalize_steps, profile_parameter_values

    conf = PipelineConf(
        description="fusion thresholds",
        config={
            "ENTITY_MATCHING_THRESHOLD": "0.99",
            "RELATION_MATCHING_THRESHOLD": "0.5",
        },
        tasks=["fusion_first_value"],
    )
    steps = normalize_steps(conf)
    task = Registry.get_task("fusion_first_value")
    catalog = bind_config_catalog(conf, [task], steps)
    assert profile_parameter_values(catalog["fusion_first_value"]) == {
        "entity_matching_threshold": 0.99,
        "relation_matching_threshold": 0.5,
    }

    defaults = PipelineConf(description="fusion defaults", tasks=["fusion_first_value"])
    default_steps = normalize_steps(defaults)
    default_catalog = bind_config_catalog(defaults, [task], default_steps)
    assert profile_parameter_values(default_catalog["fusion_first_value"]) == {
        "entity_matching_threshold": 0.5,
        "relation_matching_threshold": 0.5,
    }


def test_search_candidate_is_the_catalog_spec(tmp_path: Path):
    _rdf_tasks()
    match = Registry.get_task("match")
    exchange = Registry.get_task("exchange")
    fuse = Registry.get_task("fuse")
    tasks = [match, exchange, fuse]
    profile = ConfigurationProfile(
        name="fuse-chosen",
        definition=fuse.config_spec,
        bindings=[
            ParameterBinding(parameter=fuse.config_spec.parameters[0], value=0.8),
            ParameterBinding(parameter=fuse.config_spec.parameters[1], value=0.2),
        ],
    )
    seed = Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES)
    source = Data(path=tmp_path / "source.nt", format=DataFormat.RDF_NTRIPLES)
    pipe, conf = build_candidate(
        "candidate",
        tasks,
        {fuse.name: profile},
        seed,
        str(tmp_path / "a"),
    )
    plan = pipe.build(
        source=source,
        result=Data(path=tmp_path / "a" / "result.nt", format=DataFormat.RDF_NTRIPLES),
        stable_files=True,
        configCatalog=pipe.config_catalog,
    )
    assert conf.connect == "auto"
    assert conf.tasks[0] == "match"
    assert conf.tasks[1] == "exchange"
    assert conf.tasks[2].task_name == "fuse"
    assert plan.steps[2].parameters["entity_matching_threshold"] == 0.8
    assert plan.steps[2].parameters["relation_matching_threshold"] == 0.2

    catalog_path = tmp_path / "candidate.conf"
    catalog_path.write_text(dump_pipeline_catalog({"candidate": conf}))
    reloaded = load_pipeline_catalog(catalog_path).root["candidate"]
    reloaded_pipe = build_from_conf("candidate", reloaded, seed, str(tmp_path / "b"), tasks=tasks)
    reloaded_plan = reloaded_pipe.build(
        source=source,
        result=Data(path=tmp_path / "b" / "result.nt", format=DataFormat.RDF_NTRIPLES),
        stable_files=True,
        configCatalog=reloaded_pipe.config_catalog,
    )
    explicit = pipeline_conf_from_plan(plan, description="resolved candidate")
    explicit_pipe = build_from_conf("candidate", explicit, seed, str(tmp_path / "c"), tasks=tasks)
    explicit_plan = explicit_pipe.build(
        source=source,
        result=Data(path=tmp_path / "c" / "result.nt", format=DataFormat.RDF_NTRIPLES),
        stable_files=True,
        configCatalog=explicit_pipe.config_catalog,
    )
    assert reloaded_plan.fingerprint == plan.fingerprint == explicit_plan.fingerprint
    assert [(step.inputs, step.outputs) for step in explicit_plan.steps] == [
        (step.inputs, step.outputs) for step in plan.steps
    ]


def test_plan_fingerprint_survives_json_roundtrip(tmp_path: Path):
    _rdf_tasks()
    plan = _build_rdf(
        tmp_path,
        PipelineConf(description="auto", tasks=["match", "exchange", "fuse", "types"]),
        "json",
    )
    restored = type(plan).model_validate(json.loads(plan.model_dump_json()))
    assert restored.fingerprint == plan.fingerprint
    assert restored.steps[2].inputs["matches1"] == "exchange.output"
