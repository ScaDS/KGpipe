import pandas as pd
import pytest
import os
from typing import Sequence
from _pytest.compat import NotSetType
from itertools import permutations

from moviekg.datasets.pipe_out import load_pipe_out
from moviekg.evaluation.helpers import evaluate_stage, metrics_to_long_table_rows
from moviekg.pipelines.test_inc_msp import ssp, idfn

from moviekg.config import OUTPUT_ROOT

@pytest.mark.parametrize(
    "source_1, source_2, source_3", 
    permutations(list[str](ssp.keys()), 3),
    ids=idfn
)
def test_inc_ssp_evaluation(source_1, source_2, source_3):

    output_dir = OUTPUT_ROOT / f"{source_1}_{source_2}_{source_3}"

    pipeline_name = f"{source_1}_{source_2}_{source_3}"

    print("-" * 100)
    print(f"Evaluating {source_1}, {source_2}, {source_3}")
    print("-" * 100)

    if not output_dir.exists():
        pytest.skip(f"Pipeline output directory {output_dir} not found")

    pipe_out = load_pipe_out(output_dir)

    rows = []

    for stage in pipe_out.stages:
        print("-" * 100)
        print(f"{pipeline_name} - Stage: {stage.stage_name}")
        print("-" * 100)

        metrics = evaluate_stage(stage, is_ssp=False)
        rows.extend(metrics_to_long_table_rows(metrics, pipeline_name, stage.stage_name))
        # break # TODO remove

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_ROOT / f"{pipeline_name}_metrics.csv", index=False)
    print("saved metrics to", OUTPUT_ROOT / f"{pipeline_name}_metrics.csv")

def test_concatenate_long_table_rows():
    # glob
    rows = []
    for file in OUTPUT_ROOT.glob("*_metrics.csv"):
        if file.name == "all_metrics.csv":
            continue
        if os.path.getsize(file) < 3:
            continue
        df = pd.read_csv(file)
        rows.extend(df.to_dict(orient="records"))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_ROOT / "all_metrics.csv", index=False)
