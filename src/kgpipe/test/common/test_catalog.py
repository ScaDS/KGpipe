import csv
import io
import json
from uuid import uuid4

import pytest
from click.testing import CliRunner

from kgpipe.cli.list import list_cmd
from kgpipe.cli.registry import registry_cmd
from kgpipe.common.catalog import task_catalog, task_record
from kgpipe.common.discovery import load_optional_package, optional_packages
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType
from kgpipe.common.models import DataFormat, KgTask
from kgpipe.common.registry import Registry, RegistryCollisionError


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def _task(name: str, function, **kwargs) -> KgTask:
    return KgTask(
        name=name,
        input_spec={"source": DataFormat.RDF_NTRIPLES},
        output_spec={"output": DataFormat.RDF_NTRIPLES},
        function=function,
        **kwargs,
    )


def test_same_implementation_registers_once():
    name = _uid("same")

    def implementation(inputs, outputs):
        pass

    first = _task(name, implementation, description="first")
    second = _task(name, implementation, description="second")
    assert Registry.add_task(name, first) is first
    assert Registry.add_task(name, second) is first
    assert Registry.get_task(name) is first
    assert Registry.get_task(name.upper()) is first


def test_different_implementation_is_rejected():
    name = _uid("collide")

    def first_fn(inputs, outputs):
        pass

    def second_fn(inputs, outputs):
        pass

    Registry.add_task(name, _task(name, first_fn))
    with pytest.raises(RegistryCollisionError, match=name):
        Registry.add_task(name, _task(name, second_fn))
    assert Registry.get_task(name).function is first_fn


def test_task_name_must_match_registry_name():
    name = _uid("mismatch")

    def implementation(inputs, outputs):
        pass

    with pytest.raises(ValueError, match="does not match"):
        Registry.add_task(name, _task(_uid("other"), implementation))


def test_hand_built_tasks_register_under_their_task_name():
    import kgpipe_tasks.schema_alignment.ontology as ontology
    import kgpipe_tasks.text_processing.ner as ner

    ontology_task = Registry.get_task("extract_ontology_from_rdf")
    corenlp_task = Registry.get_task("corenlp_kbp_extraction")
    assert ontology_task is ontology.extract_ontology_from_rdf_task
    assert isinstance(ontology_task, KgTask)
    assert corenlp_task is ner.corenlp_kbp_extraction_task
    assert corenlp_task.backend == "docker"
    assert task_record(corenlp_task)["available"] is True


def test_register_decorator_rejects_raw_task_factories():
    with pytest.raises(TypeError, match="Registry.task"):
        Registry.register("task")(lambda: None)


def test_task_record_reports_formats_parameters_and_missing_dependency():
    name = _uid("meta")

    def implementation(inputs, outputs):
        pass

    task = _task(
        name,
        implementation,
        description="catalog sample",
        category="EntityMatching",
        backend="docker",
        dependencies=["json", "kgpipe_missing_catalog_dep"],
        config_spec=ConfigurationDefinition(
            name="sample",
            parameters=[
                Parameter(name="threshold", datatype=ParameterType.number, default_value=0.5),
            ],
        ),
    )
    Registry.add_task(name, task)
    record = task_record(Registry.get_task(name))
    json.dumps(record)
    assert record["inputs"] == {"source": "nt"}
    assert record["outputs"] == {"output": "nt"}
    assert record["categories"] == ["EntityMatching"]
    assert record["backend"] == "docker"
    assert record["parameters"][0]["name"] == "threshold"
    assert record["parameters"][0]["default_value"] == 0.5
    assert record["available"] is False
    assert record["missing_dependencies"] == ["kgpipe_missing_catalog_dep"]

    matched = task_catalog(category="EntityMatching", backend="docker", available=False, name=name)
    assert [item["name"] for item in matched] == [name]
    assert task_catalog(name=name, available=True) == []


def test_optional_package_status_records_import_failure():
    assert load_optional_package("catalog-json", "json") is True
    assert load_optional_package("catalog-missing", "kgpipe_missing_optional_pkg") is False
    assert load_optional_package("catalog-missing", "kgpipe_missing_optional_pkg") is False

    status = {item["name"]: item for item in optional_packages()}
    assert status["catalog-json"]["available"] is True
    assert status["catalog-missing"]["available"] is False
    assert status["catalog-missing"]["error"]
    assert sum(item["name"] == "catalog-missing" for item in optional_packages()) == 1


def test_list_and_registry_json_agree(monkeypatch):
    name = _uid("cli")

    def implementation(inputs, outputs):
        pass

    Registry.add_task(name, _task(name, implementation, category="Fusion", backend="remote"))
    monkeypatch.setattr("kgpipe.cli.list.discover_entry_points", lambda: None)
    monkeypatch.setattr("kgpipe.cli.registry.discover_entry_points", lambda: None)

    runner = CliRunner()
    listed = runner.invoke(list_cmd, ["--type", "tasks", "--format", "json", "--name", name])
    registered = runner.invoke(registry_cmd, ["--type", "task", "--format", "json", "--name", name])
    assert listed.exit_code == 0, listed.output
    assert registered.exit_code == 0, registered.output
    listed_tasks = json.loads(listed.output)["tasks"]
    registered_tasks = json.loads(registered.output)["tasks"]
    assert listed_tasks == registered_tasks
    assert listed_tasks[0]["name"] == name
    assert listed_tasks[0]["categories"] == ["Fusion"]
    assert listed_tasks[0]["backend"] == "remote"
    assert listed_tasks[0]["available"] is True

    table = runner.invoke(list_cmd, ["--type", "tasks", "--format", "table", "--name", name])
    assert table.exit_code == 0, table.output
    assert name in table.output

    csv_result = runner.invoke(list_cmd, ["--type", "tasks", "--format", "csv", "--name", name])
    assert csv_result.exit_code == 0, csv_result.output
    rows = list(csv.DictReader(io.StringIO(csv_result.output)))
    assert rows[0]["name"] == name
    assert json.loads(rows[0]["inputs"]) == {"source": "nt"}
    assert "Number of tasks" not in csv_result.output
