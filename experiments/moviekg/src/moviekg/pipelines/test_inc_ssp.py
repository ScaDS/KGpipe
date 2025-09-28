from moviekg.pipelines.helpers import run_helper, run_helper_to_json
from moviekg.config import pipeline_types, llm_pipeline_types, OUTPUT_ROOT
from pathlib import Path

import pytest



@pytest.mark.parametrize(
    "pipeline_name", 
    list(pipeline_types.keys())
)
def test_inc_ssp_classic(pipeline_name):

    print("-" * 100)
    print(f"Running {pipeline_name}")
    print("-" * 100)

    output_dir = OUTPUT_ROOT / pipeline_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_helper(pipeline_name, pipeline_types[pipeline_name], 1, 1, output_dir)
    run_helper(pipeline_name, pipeline_types[pipeline_name], 2, 2, output_dir)
    run_helper(pipeline_name, pipeline_types[pipeline_name], 3, 3, output_dir)

# @pytest.mark.skip(reason="LLM pipelines are not implemented yet")
@pytest.mark.parametrize(
    "pipeline_name", 
    list(llm_pipeline_types.keys())
)
def test_inc_ssp_with_llm(pipeline_name):
    print("-" * 100)
    print(f"Running {pipeline_name} with KG")
    print("-" * 100)

    output_dir = OUTPUT_ROOT / pipeline_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_helper(pipeline_name, llm_pipeline_types[pipeline_name], 1, 1, output_dir)
    run_helper(pipeline_name, llm_pipeline_types[pipeline_name], 2, 2, output_dir)
    run_helper(pipeline_name, llm_pipeline_types[pipeline_name], 3, 3, output_dir)

# def test_inc_ssp_to_json():
#     all_tasks_dict = []
#     for pipeline_name in list(pipeline_types.keys()):
#         tasks_dict = run_helper_to_json(pipeline_name, pipeline_types[pipeline_name], "kg")
#         all_tasks_dict.append(tasks_dict)
    
#     import json
#     json.dump(all_tasks_dict, open("all_tasks_dict.json", "w"), indent=4)