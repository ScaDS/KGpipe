from pathlib import Path

import pytest

from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType
from kgpipe.common.models import Data, DataFormat, KgTask
from kgpipe.common.registry import Registry
from kgpipe.generation.loaders import (
    build_from_conf,
    build_from_yaml,
    validate_pipeline,
)
from kgpipe.generation.spec import (
    PIPELINE_SPEC_SCHEMA,
    CatalogValidationError,
    PipelineConf,
    PipelineTaskConf,
    dump_pipeline_catalog,
    load_pipeline_catalog,
)


def _register(task: KgTask) -> KgTask:
    Registry.add_task(task.name, task)
    return task


def _rdf_tasks() -> None:
    _register(KgTask(
        name="yaml_match",
        input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
        output_spec={"output": DataFormat.PARIS_CSV},
        function=lambda _inputs, _outputs: None,
        backend="python",
    ))
    _register(KgTask(
        name="yaml_exchange",
        input_spec={"input": DataFormat.PARIS_CSV},
        output_spec={"output": DataFormat.ER_JSON},
        function=lambda _inputs, _outputs: None,
        backend="python",
    ))
    _register(KgTask(
        name="yaml_fuse",
        input_spec={
            "source": DataFormat.RDF_NTRIPLES,
            "kg": DataFormat.RDF_NTRIPLES,
            "matches1": DataFormat.ER_JSON,
        },
        output_spec={"output": DataFormat.RDF_NTRIPLES},
        function=lambda _inputs, _outputs: None,
        backend="python",
        config_spec=ConfigurationDefinition(
            name="yaml-fuse-config",
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
        name="yaml_types",
        input_spec={"source": DataFormat.RDF_NTRIPLES},
        output_spec={"result": DataFormat.RDF_NTRIPLES},
        function=lambda _inputs, _outputs: None,
        backend="python",
    ))
    _register(KgTask(
        name="yaml_json_only",
        input_spec={"input": DataFormat.JSON},
        output_spec={"output": DataFormat.JSON},
        function=lambda _inputs, _outputs: None,
        backend="docker",
    ))


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_env_defaults_and_relative_paths(tmp_path: Path):
    _rdf_tasks()
    catalog_dir = tmp_path / "catalog"
    data_dir = catalog_dir / "data"
    data_dir.mkdir(parents=True)
    yaml_path = _write(
        catalog_dir / "pipeline.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "env:",
            "  KGPIPE_THRESHOLD: \"0.75\"",
            "defaults:",
            "  data_dir: ./work",
            "  source: ./data/source.nt",
            "  seed: ${SEED_FILE}",
            "  result: ./work/result.nt",
            "rdf:",
            "  description: env sample",
            "  defaults:",
            "    data_dir: ./pipeline-work",
            "  parameters:",
            "    entity_matching_threshold: ${KGPIPE_THRESHOLD}",
            "  tasks:",
            "    - yaml_match",
            "    - yaml_exchange",
            "    - yaml_fuse",
            "    - yaml_types",
            "",
        ]),
    )
    seed = tmp_path / "outside" / "seed.nt"
    seed.parent.mkdir()
    loaded = load_pipeline_catalog(
        yaml_path,
        environ={"SEED_FILE": str(seed), "KGPIPE_THRESHOLD": "0.75"},
    )
    defaults = loaded.resolved_defaults("rdf")
    assert Path(defaults.source) == (data_dir / "source.nt").resolve()
    assert Path(defaults.seed) == seed
    assert Path(defaults.data_dir) == (catalog_dir / "pipeline-work").resolve()
    assert float(loaded.root["rdf"].parameters["entity_matching_threshold"]) == 0.75

    pipe = build_from_yaml(
        yaml_path,
        "rdf",
        environ={"SEED_FILE": str(seed), "KGPIPE_THRESHOLD": "0.75"},
    )
    assert pipe.plan.steps[2].parameters["entity_matching_threshold"] == 0.75
    assert pipe.seed.path == seed
    assert Path(pipe.data_dir) == (catalog_dir / "pipeline-work").resolve()


def test_missing_env_var_is_actionable(tmp_path: Path):
    path = _write(
        tmp_path / "pipeline.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "missing env"',
            "  defaults:",
            "    source: ${MISSING_SOURCE}",
            "  tasks:",
            "    - match",
            "",
        ]),
    )
    with pytest.raises(CatalogValidationError, match="MISSING_SOURCE"):
        load_pipeline_catalog(path, environ={})


def test_env_default_placeholder(tmp_path: Path):
    _rdf_tasks()
    path = _write(
        tmp_path / "pipeline.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "default env"',
            "  parameters:",
            "    unused: ${UNSET_VALUE:-ok}",
            "  tasks:",
            "    - yaml_match",
            "",
        ]),
    )
    # `unused` will fail later as a parameter; interpolation must succeed first.
    with pytest.raises(CatalogValidationError, match="did not match any task"):
        build_from_yaml(path, environ={})


def test_port_refs_are_not_env_interpolation(tmp_path: Path):
    _rdf_tasks()
    path = _write(
        tmp_path / "pipeline.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "explicit ports"',
            "  connect: explicit",
            "  tasks:",
            "    - id: match",
            "      task: yaml_match",
            "      inputs: {source: $source, kg: $seed}",
            "      outputs: {output: match.output}",
            "",
        ]),
    )
    catalog = load_pipeline_catalog(path, environ={})
    step = catalog.root["demo"].tasks[0]
    assert step.inputs == {"source": "$source", "kg": "$seed"}


def test_unexpected_field_error_names_pipeline(tmp_path: Path):
    path = _write(
        tmp_path / "pipeline.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "extra"',
            "  tasks:",
            "    - match",
            "  not_a_field: true",
            "",
        ]),
    )
    with pytest.raises(CatalogValidationError, match="pipeline 'demo'"):
        load_pipeline_catalog(path)
    with pytest.raises(CatalogValidationError, match="not_a_field"):
        load_pipeline_catalog(path)


def test_unknown_task_and_backend_mismatch(tmp_path: Path):
    _rdf_tasks()
    unknown = _write(
        tmp_path / "unknown.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "unknown task"',
            "  tasks:",
            "    - definitely_missing_task",
            "",
        ]),
    )
    with pytest.raises(CatalogValidationError, match="definitely_missing_task"):
        validate_pipeline(unknown)

    mismatch = _write(
        tmp_path / "backend.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "backend mismatch"',
            "  backend: docker",
            "  tasks:",
            "    - yaml_match",
            "",
        ]),
    )
    with pytest.raises(CatalogValidationError, match="backend 'docker'"):
        validate_pipeline(mismatch)

    unknown_backend = _write(
        tmp_path / "bad-backend.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "bad backend"',
            "  tasks:",
            "    - name: yaml_match",
            "      backend: spark",
            "",
        ]),
    )
    with pytest.raises(CatalogValidationError, match="Unknown backend"):
        load_pipeline_catalog(unknown_backend)


def test_invalid_task_combination_rejected_before_run(tmp_path: Path):
    _rdf_tasks()
    path = _write(
        tmp_path / "bad-combo.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "demo:",
            '  description: "incompatible tasks"',
            "  tasks:",
            "    - yaml_json_only",
            "    - yaml_types",
            "",
        ]),
    )
    with pytest.raises(CatalogValidationError, match="Cannot connect task"):
        validate_pipeline(path)


def test_yaml_and_python_plans_are_equivalent(tmp_path: Path):
    _rdf_tasks()
    yaml_path = _write(
        tmp_path / "pipeline.yaml",
        "\n".join([
            f"schema: {PIPELINE_SPEC_SCHEMA}",
            "rdf:",
            '  description: "from yaml"',
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
            "      task: yaml_match",
            "      backend: python",
            "      backend_options: {timeout: 30}",
            "    - yaml_exchange",
            "    - yaml_fuse",
            "    - yaml_types",
            "",
        ]),
    )
    python_conf = PipelineConf(
        description="from python",
        connect="auto",
        backend="python",
        parameters={"entity_matching_threshold": 0.99},
        tasks=[
            PipelineTaskConf(id="match", task="yaml_match", backend="python", backend_options={"timeout": 30}),
            "yaml_exchange",
            "yaml_fuse",
            "yaml_types",
        ],
    )
    source = tmp_path / "source.nt"
    seed = tmp_path / "seed.nt"
    yaml_pipe = build_from_yaml(
        yaml_path,
        source=source,
        seed=seed,
        result=tmp_path / "yaml" / "result.nt",
        data_dir=tmp_path / "yaml",
        environ={},
    )
    python_pipe = build_from_conf("rdf", python_conf, yaml_pipe.seed, str(tmp_path / "python"))
    python_plan = python_pipe.build(
        source=yaml_pipe.plan.source,
        result=Data(path=tmp_path / "python" / "result.nt", format=DataFormat.RDF_NTRIPLES),
        stable_files=True,
    )
    assert yaml_pipe.plan.fingerprint == python_plan.fingerprint
    assert [(step.inputs, step.outputs, step.parameters) for step in yaml_pipe.plan.steps] == [
        (step.inputs, step.outputs, step.parameters) for step in python_plan.steps
    ]
    assert yaml_pipe.step_specs[0].backend == "python"
    assert yaml_pipe.step_specs[0].backend_options == {"timeout": 30}
    assert yaml_pipe.step_specs[1].backend == "python"


def test_example_sample_yaml_loads():
    repo = Path(__file__).resolve().parents[4]
    sample = repo / "experiments/examples/sample.yaml"
    catalog = load_pipeline_catalog(sample, environ={"KGPIPE_THRESHOLD": "0.99"})
    assert catalog.schema_version == PIPELINE_SPEC_SCHEMA
    assert list(catalog.root) == ["sample"]
    conf = catalog.root["sample"]
    assert conf.backend == "python"
    assert conf.tasks[0].id == "match"
    assert conf.tasks[0].backend == "python"
    assert float(conf.parameters["entity_matching_threshold"]) == 0.99
    dumped = dump_pipeline_catalog(catalog)
    assert "schema: kgpipe.pipeline/v1" in dumped


def test_moviekg_rdf_base_migration_still_loads():
    repo = Path(__file__).resolve().parents[4]
    catalog = load_pipeline_catalog(repo / "experiments/moviekg/pipeline.conf")
    rdf = catalog.root["rdf_base"]
    assert catalog.schema_version == PIPELINE_SPEC_SCHEMA
    assert rdf.parameters["entity_matching_threshold"] == 0.99
    assert rdf.connect == "auto"
    assert "paris_entity_matching" in rdf.tasks
