from __future__ import annotations

import json
from pathlib import Path

from kgpipe_eval.utils.metric_utils import eval_results_jsons_to_rows, write_eval_csv


def test_eval_results_json_to_rows_and_csv(tmp_path: Path) -> None:
    # Create a fake output structure: <root>/<pipeline>/stage_1/eval_results.json
    p = tmp_path / "rdf_a" / "stage_1"
    p.mkdir(parents=True)

    (p / "eval_results.json").write_text(
        json.dumps(
            [
                {
                    "metric": "DuplicateMetric",
                    "summary": "Duplicates in the KG",
                    "measurements": [
                        {"name": "duplicates", "value": 3, "unit": "number"},
                        {"name": "entity_count", "value": 10, "unit": "number"},
                        {"name": "duplicates_ratio", "value": 0.3, "unit": "percentage"},
                    ],
                }
            ]
        )
    )

    allowlist = {
        "DuplicateMetric": {
            "duplicates": "number",
            "entity_count": "number",
            "duplicates_ratio": "percentage",
        }
    }

    rows = eval_results_jsons_to_rows([p / "eval_results.json"], allowlist=allowlist)
    assert rows == [
        {
            "pipeline": "rdf_a",
            "stage": "stage_1",
            "DuplicateMetric__duplicates__number": 3,
            "DuplicateMetric__entity_count__number": 10,
            "DuplicateMetric__duplicates_ratio__percentage": 0.3,
        }
    ]

    out_csv = tmp_path / "out.csv"
    write_eval_csv([p / "eval_results.json"], out_path=out_csv, allowlist=allowlist)
    txt = out_csv.read_text()

    # Header + one row, with stable columns including pipeline/stage and allowlist columns.
    lines = [l for l in txt.splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0].split(",") == [
        "pipeline",
        "stage",
        "DuplicateMetric__duplicates__number",
        "DuplicateMetric__duplicates_ratio__percentage",
        "DuplicateMetric__entity_count__number",
    ]
    assert lines[1].split(",") == ["rdf_a", "stage_1", "3", "0.3", "10"]

