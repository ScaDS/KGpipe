from __future__ import annotations

from typing import Any, Dict, Tuple

from kgpipe_search.configuration import (
    _task_categories_list,
    enumerate_valid_task_combinations,
    pipeline_config_from_snapshot,
)
from kgpipe_search.definitions import PipelineLayout, task_dict


def validate_pipeline_config_snapshot(
    snapshot: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
) -> Tuple[bool, str]:
    """
    Validate an LLM-produced pipeline config snapshot against the search space and layout.

    Returns (is_valid, error_message). error_message is empty when valid.
    """
    task_keys = snapshot.get("task_keys")
    if not isinstance(task_keys, list) or not task_keys:
        return False, "snapshot must contain a non-empty task_keys list"
    if not all(isinstance(key, str) for key in task_keys):
        return False, "task_keys must contain only strings"

    unknown = [key for key in task_keys if key not in search_space]
    if unknown:
        return False, f"unknown task keys: {unknown}"

    valid_combos = {
        tuple(combo)
        for combo in enumerate_valid_task_combinations(search_space, pipeline_layout)
    }
    if tuple(task_keys) not in valid_combos:
        return False, f"task_keys {task_keys!r} is not a valid implementation assignment"

    covered: set[str] = set()
    for task_key in task_keys:
        covered.update(_task_categories_list(search_space, task_key))

    required = set(pipeline_layout.allowed_task_categories)
    if not required.issubset(covered):
        missing = sorted(required - covered)
        return False, f"pipeline does not cover required categories: {missing}"

    profiles = snapshot.get("profiles")
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, dict):
        return False, "profiles must be an object when present"

    for task_key in task_keys:
        task = task_dict[task_key]
        task_space = search_space[task_key]
        param_names = [
            name
            for name, values in task_space.items()
            if name != "category" and isinstance(values, list)
        ]

        if not param_names:
            continue

        if getattr(task, "config_spec", None) is None:
            continue

        profile = profiles.get(task.name)
        if profile is None:
            return False, f"missing profile for task {task.name!r}"

        bindings = profile.get("bindings")
        if not isinstance(bindings, list):
            return False, f"profile for {task.name!r} must have bindings list"

        binding_map: Dict[str, Any] = {}
        for binding in bindings:
            if not isinstance(binding, dict):
                return False, f"invalid binding entry for {task.name!r}"
            param = binding.get("parameter")
            value = binding.get("value")
            if not isinstance(param, str):
                return False, f"binding parameter must be a string for {task.name!r}"
            binding_map[param] = value

        for param_name in param_names:
            allowed = task_space[param_name]
            if param_name not in binding_map:
                return False, f"missing parameter {param_name!r} for task {task_key!r}"
            if binding_map[param_name] not in allowed:
                return False, (
                    f"invalid value for {task_key!r}.{param_name}: "
                    f"{binding_map[param_name]!r} not in {allowed!r}"
                )

        extra = set(binding_map) - set(param_names)
        if extra:
            return False, f"unexpected parameters for {task_key!r}: {sorted(extra)}"

    try:
        pipeline_config_from_snapshot(snapshot)
    except Exception as exc:  # noqa: BLE001 - surface parse errors to caller
        return False, f"failed to build pipeline config: {exc}"

    return True, ""


def search_space_description(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
) -> Dict[str, Any]:
    """Serialize search space and layout for LLM prompts."""
    tasks: Dict[str, Any] = {}
    for task_key, task_space in search_space.items():
        entry: Dict[str, Any] = {"category": task_space.get("category")}
        for name, values in task_space.items():
            if name == "category":
                continue
            if isinstance(values, list):
                entry[name] = values
        tasks[task_key] = entry

    valid_combos = enumerate_valid_task_combinations(search_space, pipeline_layout)
    return {
        "pipeline_layout": {
            "allowed_task_categories": pipeline_layout.allowed_task_categories,
        },
        "tasks": tasks,
        "valid_task_combinations": valid_combos,
        "output_schema": {
            "task_keys": ["<task_key>", "..."],
            "profiles": {
                "<task_name>": {
                    "profile_name": "<optional descriptive name>",
                    "bindings": [{"parameter": "<param>", "value": "<allowed value>"}],
                }
            },
        },
    }
