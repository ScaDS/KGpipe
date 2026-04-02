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
                    parameter=Parameter(
                        name="some_parameter",
                        native_keys=["some_parameter"],
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