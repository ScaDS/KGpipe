from pathlib import Path

from kgpipe.common import Data


def test_python_task_execution(tmp_path: Path):
    from kgpipe_examples.config import ExtendedFormats
    from kgpipe_examples.task_examples import pipe_task_python

    in_file = tmp_path / "input.special_in"
    out_file = tmp_path / "output.special1"
    in_file.touch()

    report = pipe_task_python.run(
        inputs=[Data(path=in_file, format=ExtendedFormats.SPECIAL_IN)],
        outputs=[Data(path=out_file, format=ExtendedFormats.SPECIAL1)],
    )

    assert pipe_task_python.name == "pipe_task_python"
    assert report.status == "success"
    assert out_file.exists()


def test_docker_task_execution(tmp_path: Path):
    from kgpipe_examples.config import ExtendedFormats
    from kgpipe_examples.task_examples import pipe_task_docker

    in_file = tmp_path / "input.special1"
    out_file = tmp_path / "output.special2"
    in_file.touch()

    report = pipe_task_docker.run(
        inputs=[Data(path=in_file, format=ExtendedFormats.SPECIAL1)],
        outputs=[Data(path=out_file, format=ExtendedFormats.SPECIAL2)],
    )

    assert pipe_task_docker.name == "pipe_task_docker"
    assert report.status == "success"
    assert out_file.exists()


def test_remote_task_execution(tmp_path: Path):
    from kgpipe_examples.config import ExtendedFormats
    from kgpipe_examples.task_examples import pipe_task_remote

    in_file = tmp_path / "input.special2"
    out_file = tmp_path / "output.special_kg"
    in_file.touch()

    report = pipe_task_remote.run(
        inputs=[Data(path=in_file, format=ExtendedFormats.SPECIAL2)],
        outputs=[Data(path=out_file, format=ExtendedFormats.SPECIAL_KG)],
    )

    assert pipe_task_remote.name == "pipe_task_remote"
    assert report.status == "success"
    assert out_file.exists()


def test_config_spec_execution(tmp_path: Path):
    from kgpipe_examples.config import ExtendedFormats
    from kgpipe_examples.task_examples import pipe_task_with_config
    from kgpipe.common.model.configuration import (
        ConfigurationProfile,
        ParameterBinding,
        Parameter,
        ParameterType,
    )

    in_file = tmp_path / "input.special1"
    out_file = tmp_path / "output.special_kg"
    in_file.touch()
    
    report = pipe_task_with_config.run(
        inputs=[Data(path=in_file, format=ExtendedFormats.SPECIAL1)],
        outputs=[Data(path=out_file, format=ExtendedFormats.SPECIAL_KG)],
        configProfile=ConfigurationProfile(
            name="pipe_task_with_config_profile",
            definition=pipe_task_with_config.config_spec,
            bindings=[
                ParameterBinding(
                    parameter=next(
                        p for p in pipe_task_with_config.config_spec.parameters
                        if p.name == "some_parameter"
                    ),
                    value="some",
                )
            ],
        ),
    )

    assert pipe_task_with_config.name == "pipe_task_with_config"
    assert report.status == "success"
    assert out_file.exists()

def test_config_profile_missing_fails(tmp_path: Path):
    from kgpipe_examples.config import ExtendedFormats
    from kgpipe_examples.task_examples import pipe_task_with_config

    in_file = tmp_path / "input.special1"
    out_file = tmp_path / "output.special_kg"
    in_file.touch()

    report = pipe_task_with_config.run(
        inputs=[Data(path=in_file, format=ExtendedFormats.SPECIAL1)],
        outputs=[Data(path=out_file, format=ExtendedFormats.SPECIAL_KG)],
        # configProfile intentionally omitted
    )

    assert report.status == "failed"
    assert report.error is not None
    assert "requires a 'config' argument" in report.error


def test_config_profile_wrong_type_fails(tmp_path: Path):
    from kgpipe_examples.config import ExtendedFormats
    from kgpipe_examples.task_examples import pipe_task_with_config

    in_file = tmp_path / "input.special1"
    out_file = tmp_path / "output.special_kg"
    in_file.touch()

    report = pipe_task_with_config.run(
        inputs=[Data(path=in_file, format=ExtendedFormats.SPECIAL1)],
        outputs=[Data(path=out_file, format=ExtendedFormats.SPECIAL_KG)],
        configProfile="not-a-profile",
    )

    assert report.status == "failed"
    assert report.error is not None
    assert "expects configProfile to be a ConfigurationProfile" in report.error


def test_config_profile_spec_mismatch_fails(tmp_path: Path):
    from kgpipe_examples.config import ExtendedFormats
    from kgpipe_examples.task_examples import pipe_task_with_config
    from kgpipe.common.model.configuration import (
        ConfigurationProfile,
        ConfigurationDefinition,
    )

    in_file = tmp_path / "input.special1"
    out_file = tmp_path / "output.special_kg"
    in_file.touch()

    report = pipe_task_with_config.run(
        inputs=[Data(path=in_file, format=ExtendedFormats.SPECIAL1)],
        outputs=[Data(path=out_file, format=ExtendedFormats.SPECIAL_KG)],
        configProfile=ConfigurationProfile(
            name="mismatching_profile",
            definition=ConfigurationDefinition(name="different_spec_name"),
            bindings=[],
        ),
    )

    assert report.status == "failed"
    assert report.error is not None
    assert "does not match task config spec" in report.error


def test_config_profile_unknown_parameter_fails(tmp_path: Path):
    from kgpipe_examples.config import ExtendedFormats
    from kgpipe_examples.task_examples import pipe_task_with_config
    from kgpipe.common.model.configuration import (
        ConfigurationProfile,
        ParameterBinding,
        Parameter,
        ParameterType,
    )

    in_file = tmp_path / "input.special1"
    out_file = tmp_path / "output.special_kg"
    in_file.touch()

    report = pipe_task_with_config.run(
        inputs=[Data(path=in_file, format=ExtendedFormats.SPECIAL1)],
        outputs=[Data(path=out_file, format=ExtendedFormats.SPECIAL_KG)],
        configProfile=ConfigurationProfile(
            name="pipe_task_with_config_profile_unknown_param",
            definition=pipe_task_with_config.config_spec,
            bindings=[
                ParameterBinding(
                    parameter=Parameter(
                        name="other_parameter",
                        native_keys=["other_parameter"],
                        datatype=ParameterType.string,
                        default_value="default",
                        required=False,
                        allowed_values=[],
                    ),
                    value="some",
                )
            ],
        ),
    )

    assert report.status == "failed"
    assert report.error is not None
    assert "Unknown config parameter" in report.error

def test_pipeline_definition_executes():
    from kgpipe_examples.pipe_examples import pipe_example

    # Main objective: execute the pipeline example end-to-end without errors.
    pipe_example()

def test_evaluation_example(tmp_path: Path):
    from kgpipe_examples.eval_examples import eval_example
    eval_example(tmp_path)


def test_example_pipeline_catalog_resolves_ports(tmp_path: Path):
    from kgpipe.common.models import Data, DataFormat
    from kgpipe.generation.loaders import build_from_conf, load_pipeline_catalog
    import kgpipe_examples.dummy_tasks  # noqa: F401

    catalog = load_pipeline_catalog(Path(__file__).resolve().parents[2] / "pipeline.conf")
    seed = Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES)
    source = Data(path=tmp_path / "source.nt", format=DataFormat.RDF_NTRIPLES)
    seed.path.touch()
    source.path.touch()

    def build(name: str):
        result = Data(path=tmp_path / name / "result.nt", format=DataFormat.RDF_NTRIPLES)
        pipe = build_from_conf(name, catalog.root[name], seed, str(tmp_path / name))
        return pipe.build(source=source, result=result, stable_files=True)

    auto = build("dummy_auto")
    explicit = build("dummy_explicit")
    assert [(step.inputs, step.outputs) for step in auto.steps] == [
        ({"source": "$source", "kg": "$seed"}, {"output": "dummy_match.output"}),
        ({"input": "dummy_match.output"}, {"output": "dummy_exchange.output"}),
        (
            {"source": "$source", "kg": "$seed", "matches1": "dummy_exchange.output"},
            {"output": "dummy_fuse.output"},
        ),
        ({"source": "dummy_fuse.output"}, {"result": "$result"}),
    ]
    assert auto.steps[0].parameters["entity_matching_threshold"] == 0.99
    assert auto.steps[2].parameters["entity_matching_threshold"] == 0.7
    assert explicit.fingerprint == auto.fingerprint

    join_auto = build("dummy_join_auto")
    join_explicit = build("dummy_join_explicit")
    assert join_auto.steps[2].inputs == {
        "json1": "dummy_match_2.output",
        "json2": "dummy_match.output",
    }
    assert join_explicit.steps[2].inputs == {
        "json1": "entities.output",
        "json2": "relations.output",
    }
    assert join_explicit.fingerprint != join_auto.fingerprint


def test_sample_yaml_matches_python_plan(tmp_path: Path):
    from kgpipe.common.models import Data, DataFormat
    from kgpipe.generation.loaders import build_from_conf, build_from_yaml, load_pipeline_catalog
    import kgpipe_examples.dummy_tasks  # noqa: F401

    sample = Path(__file__).resolve().parents[2] / "sample.yaml"
    catalog = load_pipeline_catalog(sample, environ={"KGPIPE_THRESHOLD": "0.99"})
    conf = catalog.root["sample"]
    python_pipe = build_from_conf(
        "sample",
        conf,
        Data(path=tmp_path / "seed.nt", format=DataFormat.RDF_NTRIPLES),
        str(tmp_path / "python"),
    )
    python_plan = python_pipe.build(
        source=Data(path=tmp_path / "source.nt", format=DataFormat.RDF_NTRIPLES),
        result=Data(path=tmp_path / "python" / "result.nt", format=DataFormat.RDF_NTRIPLES),
        stable_files=True,
    )
    yaml_pipe = build_from_yaml(
        sample,
        source=tmp_path / "source.nt",
        seed=tmp_path / "seed.nt",
        result=tmp_path / "yaml" / "result.nt",
        data_dir=tmp_path / "yaml",
        environ={"KGPIPE_THRESHOLD": "0.99"},
    )
    assert yaml_pipe.plan.fingerprint == python_plan.fingerprint
    assert yaml_pipe.plan.steps[0].parameters["entity_matching_threshold"] == 0.8
    assert yaml_pipe.plan.steps[2].parameters["entity_matching_threshold"] == 0.99


def test_public_api_matches_sample_yaml(tmp_path: Path):
    from kgpipe import Pipeline
    import kgpipe_examples.dummy_tasks  # noqa: F401
    from kgpipe_examples.api_example import sample_pipeline

    sample = Path(__file__).resolve().parents[2] / "sample.yaml"
    source = tmp_path / "source.nt"
    seed = tmp_path / "seed.nt"
    yaml_pipeline = Pipeline.from_yaml(
        sample,
        source=source,
        seed=seed,
        result=tmp_path / "yaml" / "result.nt",
        data_dir=tmp_path / "yaml",
        environ={"KGPIPE_THRESHOLD": "0.99"},
    )
    python_pipeline = sample_pipeline(
        source,
        seed,
        tmp_path / "python" / "result.nt",
        tmp_path / "python",
    )
    assert python_pipeline.plan().fingerprint == yaml_pipeline.plan().fingerprint
    assert python_pipeline.plan().steps[0].parameters["entity_matching_threshold"] == 0.8
    assert python_pipeline.plan().steps[2].parameters["entity_matching_threshold"] == 0.99
