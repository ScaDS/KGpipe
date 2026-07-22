from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from kgpipe_search.configuration import (
    pipeline_config_from_snapshot,
    pipeline_config_snapshot_key,
    pipeline_config_to_snapshot,
    sample_valid_pipeline_config,
    task_keys_from_pipeline_config,
)
from kgpipe_search.definitions import PipelineConfig, PipelineLayout
from kgpipe_search.strategies.llm_client import ChatCompletionClient, OpenAICompatibleClient
from kgpipe_search.strategies.llm_validation import (
    search_space_description,
    validate_pipeline_config_snapshot,
)
from kgpipe_search.strategies.strategies import EvaluateFn, Observation, SearchRun

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    for match in _JSON_BLOCK_RE.finditer(text):
        candidates.append(match.group(1).strip())

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    raise ValueError(f"Could not parse JSON object from LLM response: {text!r}") from last_error


def _history_summary(history: List[Observation]) -> List[Dict[str, Any]]:
    ranked = sorted(history, key=lambda item: item[0], reverse=True)
    summary: List[Dict[str, Any]] = []
    for score, cfg in ranked[:5]:
        task_keys = task_keys_from_pipeline_config(cfg)
        summary.append(
            {
                "score": score,
                "snapshot": pipeline_config_to_snapshot(task_keys, cfg),
            }
        )
    return summary


def _build_prompt(
    *,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    history: List[Observation],
    evaluated_keys: Set[str],
    attempt: int,
    last_error: str,
) -> Tuple[str, str]:
    system = (
        "You propose valid KGpipe pipeline configurations. "
        "Respond with a single JSON object only. "
        "Every task_key must be chosen from valid_task_combinations. "
        "Every parameter value must be one of the allowed values in tasks."
    )
    payload = {
        "search_space": search_space_description(search_space, pipeline_layout),
        "attempt": attempt,
        "already_evaluated_count": len(evaluated_keys),
        "best_observations": _history_summary(history),
        "last_validation_error": last_error or None,
        "instructions": [
            "Pick one valid task_keys combination from valid_task_combinations.",
            "For each selected task with parameters, provide bindings using only allowed values.",
            "Prefer configs that differ from already evaluated ones when possible.",
            "Return JSON matching output_schema.",
        ],
    }
    return system, json.dumps(payload, indent=2)


def propose_pipeline_config_with_llm(
    *,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    client: ChatCompletionClient,
    history: Optional[List[Observation]] = None,
    evaluated_keys: Optional[Set[str]] = None,
    max_retries: int = 3,
) -> Tuple[PipelineConfig, str]:
    """
    Ask an LLM for a pipeline config snapshot, validate it, and retry on failure.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    observations = history or []
    seen = evaluated_keys or set()
    last_error = ""

    for attempt in range(1, max_retries + 1):
        system, user = _build_prompt(
            search_space=search_space,
            pipeline_layout=pipeline_layout,
            history=observations,
            evaluated_keys=seen,
            attempt=attempt,
            last_error=last_error,
        )
        raw = client.complete(system=system, user=user)
        try:
            snapshot = _extract_json_object(raw)
        except ValueError as exc:
            last_error = str(exc)
            continue

        is_valid, error = validate_pipeline_config_snapshot(
            snapshot, search_space, pipeline_layout
        )
        if not is_valid:
            last_error = error
            continue

        config = pipeline_config_from_snapshot(snapshot)
        key = pipeline_config_snapshot_key(config, search_space)
        if key in seen:
            last_error = "configuration was already evaluated"
            continue

        return config, f"llm(attempt={attempt})"

    raise RuntimeError(
        f"LLM failed to produce a valid unevaluated config after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def run_llm(
    *,
    budget: int,
    evaluate_fn: EvaluateFn,
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
    max_retries: int = 3,
    client: Optional[ChatCompletionClient] = None,
    rng: Optional[random.Random] = None,
) -> SearchRun:
    if budget <= 0:
        return SearchRun(strategy="llm", history=[], budget=0, decisions=[])

    draw = rng or random.Random()
    llm_client = client or OpenAICompatibleClient.from_env()
    history: List[Observation] = []
    decisions: List[str] = []
    evaluated_keys: Set[str] = set()

    for _ in range(budget):
        try:
            candidate, decision = propose_pipeline_config_with_llm(
                search_space=search_space,
                pipeline_layout=pipeline_layout,
                client=llm_client,
                history=history,
                evaluated_keys=evaluated_keys,
                max_retries=max_retries,
            )
        except RuntimeError:
            candidate = sample_valid_pipeline_config(search_space, pipeline_layout, rng=draw)
            decision = "fallback(random)"

        key = pipeline_config_snapshot_key(candidate, search_space)
        if key in evaluated_keys:
            candidate = sample_valid_pipeline_config(search_space, pipeline_layout, rng=draw)
            decision = f"{decision}+dedupe(random)"

        key = pipeline_config_snapshot_key(candidate, search_space)
        score = evaluate_fn(candidate)
        history.append((score, candidate))
        evaluated_keys.add(key)
        decisions.append(decision)

    return SearchRun(strategy="llm", history=history, budget=budget, decisions=decisions)
