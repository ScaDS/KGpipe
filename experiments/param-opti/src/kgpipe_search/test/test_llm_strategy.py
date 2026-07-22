import json
import random
from dataclasses import dataclass
from typing import List

from kgpipe_search.definitions import RDF_PIPELINE_LAYOUT, RDF_SEARCH_SPACE
from kgpipe_search.evaluation import execute_and_dummy_evaluate_pipeline
from kgpipe_search.search import llm_search
from kgpipe_search.strategies.llm_strategy import propose_pipeline_config_with_llm
from kgpipe_search.strategies.llm_validation import validate_pipeline_config_snapshot


@dataclass
class ScriptedLlmClient:
    responses: List[str]
    calls: int = 0

    def complete(self, *, system: str, user: str) -> str:
        del system, user
        if self.calls >= len(self.responses):
            raise RuntimeError("no more scripted responses")
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _valid_snapshot(
    *,
    entity_threshold: float = 0.7,
    relation_threshold: float = 0.6,
) -> dict:
    return {
        "task_keys": ["paris_graph_alignment_task", "fusion_first_value_task"],
        "profiles": {
            "paris_graph_alignment_task": {
                "profile_name": (
                    "paris_graph_alignment_entity_matching_threshold="
                    f"{entity_threshold},relation_matching_threshold={relation_threshold}"
                ),
                "bindings": [
                    {"parameter": "entity_matching_threshold", "value": entity_threshold},
                    {"parameter": "relation_matching_threshold", "value": relation_threshold},
                ],
            }
        },
    }


def test_validate_pipeline_config_snapshot_accepts_valid_config():
    snapshot = _valid_snapshot()
    is_valid, error = validate_pipeline_config_snapshot(
        snapshot, RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT
    )
    assert is_valid, error


def test_validate_pipeline_config_snapshot_rejects_invalid_parameter():
    snapshot = _valid_snapshot()
    snapshot["profiles"]["paris_graph_alignment_task"]["bindings"][0]["value"] = 0.42

    is_valid, error = validate_pipeline_config_snapshot(
        snapshot, RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT
    )
    assert not is_valid
    assert "invalid value" in error


def test_propose_pipeline_config_with_llm_retries_until_valid():
    invalid = {"task_keys": ["not_a_real_task"]}
    client = ScriptedLlmClient(
        responses=[
            "not json",
            json.dumps(invalid),
            json.dumps(_valid_snapshot()),
        ]
    )

    config, decision = propose_pipeline_config_with_llm(
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        client=client,
        max_retries=3,
    )

    assert client.calls == 3
    assert decision == "llm(attempt=3)"
    assert [task.name for task in config.tasks] == [
        "paris_graph_alignment_task",
        "fusion_first_value_task",
    ]


def test_llm_search_with_mocked_client():
    client = ScriptedLlmClient(
        responses=[
            json.dumps(_valid_snapshot(entity_threshold=0.7, relation_threshold=0.6)),
            json.dumps(_valid_snapshot(entity_threshold=0.8, relation_threshold=0.5)),
            json.dumps(_valid_snapshot(entity_threshold=0.9, relation_threshold=0.7)),
        ]
    )
    run = llm_search(
        budget=3,
        evaluate_fn=execute_and_dummy_evaluate_pipeline,
        search_space=RDF_SEARCH_SPACE,
        pipeline_layout=RDF_PIPELINE_LAYOUT,
        client=client,
        rng=random.Random(0),
    )

    assert len(run.history) == 3
    assert all(decision.startswith("llm(") for decision in run.decisions)
