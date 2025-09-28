from moviekg.datasets.pipe_out import load_pipe_out
from moviekg.evaluation.helpers import evaluate_stage, metrics_to_long_table_rows, print_long_table_rows
from moviekg.pipelines.test_inc_ssp import pipeline_types, llm_pipeline_types

import pytest
from pathlib import Path
import pandas as pd
import os

from moviekg.config import OUTPUT_ROOT
# create output root/paper if not exists

@pytest.mark.parametrize(
    "pipeline_name", 
    list(pipeline_types.keys()) + list(llm_pipeline_types.keys())
)
def test_inc_ssp_evaluation(pipeline_name):

    output_dir = OUTPUT_ROOT / pipeline_name

    print("-" * 100)
    print(f"Evaluating {pipeline_name}")
    print("-" * 100)

    if not output_dir.exists():
        pytest.skip(f"Pipeline output directory {output_dir} not found")

    pipe_out = load_pipe_out(output_dir)

    rows = []

    for stage in pipe_out.stages:
        print("-" * 100)
        print(f"{pipeline_name} - Stage: {stage.stage_name}")
        print("-" * 100)

        metrics = evaluate_stage(stage, is_ssp=True)
        new_rows = metrics_to_long_table_rows(metrics, pipeline_name, stage.stage_name)
        print_long_table_rows(new_rows)
        rows.extend(new_rows)
        # break # TODO remove this

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

from moviekg.evaluation.helpers import replace_with_dict

# def test_replace_with_dict():
#     replace_with_dict(str(OUTPUT_ROOT / "all_metrics.csv"), {

#     })