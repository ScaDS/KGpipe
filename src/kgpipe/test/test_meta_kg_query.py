import importlib.util
from pathlib import Path


def _load_query_module():
    module_path = Path(__file__).resolve().parents[2] / "kgpipe_view" / "meta_kg_query.py"
    spec = importlib.util.spec_from_file_location("meta_kg_query", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_query_tasks_implementations_maps_primary_results(monkeypatch):
    module = _load_query_module()

    calls = []

    def _fake_run_select(_endpoint_url, query):
        calls.append(query)
        return [
            {
                "task": {"value": "http://example.org/kgp#TaskA"},
                "method": {"value": "http://example.org/kgp#MethodA"},
                "implementation": {"value": "http://example.org/kgp#ImplA"},
                "tool": {"value": "http://example.org/kgp#ToolA"},
                "runtime": {"value": "python"},
                "implementationVersion": {"value": "1.0.0"},
                "commandTemplate": {"value": "python run.py"},
            }
        ]

    monkeypatch.setattr(module, "_run_select", _fake_run_select)

    frame = module.query_tasks_implementations("http://localhost:8890/sparql")

    assert len(calls) == 1
    assert frame.shape == (1, 7)
    assert frame.loc[0, "task"] == "http://example.org/kgp#TaskA"
    assert frame.loc[0, "implementation_version"] == "1.0.0"


def test_query_tasks_implementations_falls_back_when_primary_is_empty(monkeypatch):
    module = _load_query_module()

    calls = []

    def _fake_run_select(_endpoint_url, query):
        calls.append(query)
        if len(calls) == 1:
            return []
        return [{"implementation": {"value": "http://example.org/kgp#ImplB"}}]

    monkeypatch.setattr(module, "_run_select", _fake_run_select)

    frame = module.query_tasks_implementations("http://localhost:8890/sparql")

    assert len(calls) == 2
    assert frame.shape == (1, 7)
    assert frame.loc[0, "implementation"] == "http://example.org/kgp#ImplB"
    assert frame.loc[0, "task"] == ""


def test_query_task_hierarchy_maps_primary_results(monkeypatch):
    module = _load_query_module()

    calls = []

    def _fake_run_select(_endpoint_url, query):
        calls.append(query)
        return [
            {
                "task": {"value": "http://example.org/kgp#NormalizeTask"},
                "parentTask": {"value": "http://example.org/kgp#TransformTask"},
            }
        ]

    monkeypatch.setattr(module, "_run_select", _fake_run_select)

    frame = module.query_task_hierarchy("http://localhost:8890/sparql")

    assert len(calls) == 1
    assert frame.shape == (1, 2)
    assert frame.loc[0, "task"] == "http://example.org/kgp#NormalizeTask"
    assert frame.loc[0, "parent_task"] == "http://example.org/kgp#TransformTask"


def test_query_task_hierarchy_falls_back_when_primary_is_empty(monkeypatch):
    module = _load_query_module()

    calls = []

    def _fake_run_select(_endpoint_url, query):
        calls.append(query)
        if len(calls) == 1:
            return []
        return [{"task": {"value": "http://example.org/kgp#TrainTask"}}]

    monkeypatch.setattr(module, "_run_select", _fake_run_select)

    frame = module.query_task_hierarchy("http://localhost:8890/sparql")

    assert len(calls) == 2
    assert frame.shape == (1, 2)
    assert frame.loc[0, "task"] == "http://example.org/kgp#TrainTask"
    assert frame.loc[0, "parent_task"] == ""
