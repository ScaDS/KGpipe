#!/usr/bin/env python3
"""
Registry command for KGbench CLI.

This module handles registry management.
"""

from typing import Optional

import click

from kgpipe.cli.catalog_output import catalog_payload, emit_catalog, normalize_kind
from kgpipe.common.discovery import discover_entry_points


def _available_filter(available: bool, unavailable: bool) -> Optional[bool]:
    if available and unavailable:
        raise click.UsageError("Use only one of --available and --unavailable.")
    if available:
        return True
    if unavailable:
        return False
    return None


@click.command()
@click.option(
    "--type",
    "-t",
    "kind",
    type=str,
    help="Show only one registry type (task, pipeline, metric, evaluator)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "yaml", "csv"]),
    default="table",
    help="Output format",
)
@click.option("--name", default=None, help="Keep tasks whose name contains this text")
@click.option("--category", default=None, help="Keep tasks in this category")
@click.option("--backend", default=None, help="Keep tasks with this backend (python, docker, remote)")
@click.option("--available", is_flag=True, help="Keep tasks whose declared dependencies import")
@click.option("--unavailable", is_flag=True, help="Keep tasks with a missing declared dependency")
@click.pass_context
def registry_cmd(
    ctx: click.Context,
    kind: Optional[str],
    output_format: str,
    name: Optional[str],
    category: Optional[str],
    backend: Optional[str],
    available: bool,
    unavailable: bool,
):
    """
    Show registry contents.

    Display information about registered tasks, pipelines, metrics, and evaluators.
    JSON and YAML use the same records as ``kgpipe list``.
    """
    discover_entry_points()
    selected = normalize_kind(kind)
    payload = catalog_payload(
        selected,
        name=name,
        category=category,
        backend=backend,
        available=_available_filter(available, unavailable),
        include_packages=True,
    )
    emit_catalog(payload, output_format=output_format, kind=selected)
