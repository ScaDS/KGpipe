"""Shared table, CSV, and JSON rendering for the task catalog."""

from __future__ import annotations

import csv
import json
import sys
from typing import Any, Optional

import click
from rich.console import Console
from rich.table import Table

from kgpipe.common.catalog import records_for, task_catalog
from kgpipe.common.discovery import (
    get_registered_evaluators,
    get_registered_metrics,
    get_registered_pipelines,
    get_registered_tasks,
    optional_packages,
)

console = Console()

KIND_ALIASES = {
    "all": "all",
    "task": "tasks",
    "tasks": "tasks",
    "stage": "tasks",
    "stages": "tasks",
    "pipeline": "pipelines",
    "pipelines": "pipelines",
    "metric": "metrics",
    "metrics": "metrics",
    "evaluator": "evaluators",
    "evaluators": "evaluators",
}

TASK_COLUMNS = (
    "name",
    "categories",
    "backend",
    "available",
    "description",
    "inputs",
    "outputs",
    "parameters",
    "dependencies",
    "missing_dependencies",
    "module",
)


def normalize_kind(kind: Optional[str]) -> str:
    if not kind:
        return "all"
    try:
        return KIND_ALIASES[kind]
    except KeyError as exc:
        known = ", ".join(sorted(set(KIND_ALIASES)))
        raise click.UsageError(f"Unknown type {kind!r}. Known: {known}.") from exc


def _sources() -> dict[str, list]:
    return {
        "tasks": get_registered_tasks(),
        "pipelines": get_registered_pipelines(),
        "metrics": get_registered_metrics(),
        "evaluators": get_registered_evaluators(),
    }


def catalog_payload(
    kind: str,
    *,
    name: Optional[str] = None,
    category: Optional[str] = None,
    backend: Optional[str] = None,
    available: Optional[bool] = None,
    include_packages: bool = False,
) -> dict[str, Any]:
    sources = _sources()
    selected = list(sources) if kind == "all" else [kind]
    payload: dict[str, Any] = {}
    for item_kind in selected:
        if item_kind == "tasks":
            payload["tasks"] = task_catalog(
                sources["tasks"],
                name=name,
                category=category,
                backend=backend,
                available=available,
            )
        else:
            payload[item_kind] = records_for(item_kind, sources[item_kind])
    if include_packages and kind == "all":
        payload["packages"] = optional_packages()
    return payload


def _cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_csv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _cell(row.get(column)) for column in columns})


def _task_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Registered Tasks")
    for column, style in (
        ("Name", "cyan"),
        ("Category", "magenta"),
        ("Backend", "yellow"),
        ("Available", "green"),
        ("Description", "green"),
        ("Module", "red"),
    ):
        table.add_column(column, style=style)
    for row in rows:
        table.add_row(
            row["name"],
            ", ".join(row["categories"]),
            row["backend"],
            "yes" if row["available"] else "no",
            row["description"] or "",
            row["module"],
        )
    console.print(table)
    console.print(f"Number of tasks: {len(rows)}")


def _object_table(title: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    table = Table(title=title)
    for column in columns:
        table.add_column(column.replace("_", " ").title(), style="cyan")
    for row in rows:
        table.add_row(*(_cell(row.get(column)) for column in columns))
    console.print(table)
    console.print(f"Number of {title.lower()}: {len(rows)}")


def emit_catalog(
    payload: dict[str, Any],
    *,
    output_format: str,
    kind: str,
) -> None:
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(payload, sort_keys=False).rstrip())
        return
    if output_format == "csv":
        if kind == "all" or len(payload) != 1:
            raise click.UsageError("CSV output needs a single --type (tasks, pipelines, metrics, or evaluators).")
        item_kind, rows = next(iter(payload.items()))
        columns = TASK_COLUMNS if item_kind == "tasks" else tuple(rows[0]) if rows else ("name",)
        _write_csv(rows, columns)
        return

    if "tasks" in payload:
        _task_table(payload["tasks"])
        console.print()
    if "pipelines" in payload:
        _object_table("Registered Pipelines", payload["pipelines"], ("name", "description", "task_count", "target"))
        console.print()
    if "metrics" in payload:
        metric_rows = []
        for row in payload["metrics"]:
            labels = ", ".join(
                item["name"] + (f"[{item['unit']}]" if item.get("unit") else "")
                for item in row["measurements"]
            ) or "—"
            metric_rows.append({"key": row["key"], "description": row["description"], "measurements": labels})
        _object_table("Registered Metrics", metric_rows, ("key", "description", "measurements"))
        console.print()
    if "evaluators" in payload:
        _object_table("Registered Evaluators", payload["evaluators"], ("name", "description", "supported_metrics"))
        console.print()
    if "packages" in payload and payload["packages"]:
        _object_table("Optional Packages", payload["packages"], ("name", "import_name", "available", "error"))
