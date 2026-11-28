"""Serializable task catalog built from the registry."""

from __future__ import annotations

import importlib.util
from typing import Any, Iterable, Optional, Sequence

from kgpipe.common.models import KgTask


def _format_name(fmt: Any) -> str:
    return str(getattr(fmt, "value", fmt))


def _category_name(category: Any) -> str:
    if isinstance(category, str):
        return category
    return str(getattr(category, "name", category))


def _importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def parameter_record(parameter: Any) -> dict[str, Any]:
    datatype = getattr(parameter, "datatype", None)
    return {
        "name": parameter.name,
        "datatype": getattr(datatype, "value", datatype),
        "default_value": parameter.default_value,
        "required": bool(parameter.required),
        "allowed_values": list(parameter.allowed_values or []),
        "minimum": parameter.minimum,
        "maximum": parameter.maximum,
        "unit": parameter.unit,
        "native_keys": list(parameter.native_keys or []),
    }


def task_record(task: KgTask) -> dict[str, Any]:
    """JSON-serializable description of one registered task."""
    dependencies = list(task.dependencies)
    missing = [name for name in dependencies if not _importable(name)]
    spec = task.config_spec
    function = task.function
    module = getattr(function, "__module__", "")
    qualname = getattr(function, "__qualname__", None) or getattr(function, "__name__", "")
    return {
        "name": task.name,
        "description": task.description,
        "categories": [_category_name(category) for category in task.category],
        "inputs": {name: _format_name(fmt) for name, fmt in task.input_spec.items()},
        "outputs": {name: _format_name(fmt) for name, fmt in task.output_spec.items()},
        "parameters": [parameter_record(parameter) for parameter in (spec.parameters if spec else [])],
        "backend": task.backend,
        "dependencies": dependencies,
        "available": not missing,
        "missing_dependencies": missing,
        "module": f"{module}.{qualname}" if module else qualname,
        "see_also": list(task.see_also),
        "tools": list(task.tools),
    }


def _matches(record: dict[str, Any], *, name: Optional[str], category: Optional[str], backend: Optional[str], available: Optional[bool]) -> bool:
    if name and name.casefold() not in record["name"].casefold():
        return False
    if category:
        wanted = category.casefold()
        if wanted not in {item.casefold() for item in record["categories"]}:
            return False
    if backend and record["backend"].casefold() != backend.casefold():
        return False
    if available is not None and record["available"] is not available:
        return False
    return True


def task_catalog(
    tasks: Optional[Sequence[KgTask]] = None,
    *,
    name: Optional[str] = None,
    category: Optional[str] = None,
    backend: Optional[str] = None,
    available: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """Registered tasks as plain records, filtered and ordered by name.

    With ``tasks`` omitted, the current registry is read after the caller has
    imported or discovered task modules.
    """
    if tasks is None:
        from kgpipe.common.discovery import get_registered_tasks

        tasks = get_registered_tasks()
    records = []
    for task in tasks:
        if not isinstance(task, KgTask):
            continue
        record = task_record(task)
        if _matches(record, name=name, category=category, backend=backend, available=available):
            records.append(record)
    records.sort(key=lambda record: record["name"])
    return records


def pipeline_record(pipeline: Any) -> dict[str, Any]:
    steps = getattr(pipeline, "tasks", None)
    target = getattr(pipeline, "target", None)
    return {
        "name": getattr(pipeline, "name", None),
        "description": getattr(pipeline, "description", None),
        "task_count": len(steps) if steps is not None else None,
        "target": str(target) if target is not None else None,
    }


def metric_record(metric: Any) -> dict[str, Any]:
    measurements = []
    for measurement in getattr(metric, "measurements", ()) or ():
        alias = getattr(measurement, "alias", ()) or ()
        measurements.append({
            "name": getattr(measurement, "name", None),
            "unit": getattr(measurement, "unit", None),
            "alias": list(alias),
        })
    return {
        "key": getattr(metric, "key", getattr(metric, "__name__", None)),
        "description": getattr(metric, "description", None),
        "measurements": measurements,
    }


def evaluator_record(evaluator: Any) -> dict[str, Any]:
    supported = getattr(evaluator, "supported_metrics", []) or []
    return {
        "name": getattr(evaluator, "name", None),
        "description": getattr(evaluator, "description", None),
        "supported_metrics": [str(item) for item in supported],
    }


def records_for(kind: str, items: Iterable[Any]) -> list[dict[str, Any]]:
    if kind == "tasks":
        return task_catalog([item for item in items if isinstance(item, KgTask)])
    if kind == "pipelines":
        rows = [pipeline_record(item) for item in items]
    elif kind == "metrics":
        rows = [metric_record(item) for item in items]
    elif kind == "evaluators":
        rows = [evaluator_record(item) for item in items]
    else:
        rows = [{"name": getattr(item, "name", None) or getattr(item, "__name__", str(item))} for item in items]
    rows.sort(key=lambda row: str(row.get("name") or row.get("key") or ""))
    return rows
