from kgpipe.common import KgPipe, Data, DataFormat
from param_opti.tasks.paris import paris_entity_alignment_task, paris_graph_alignment_task
from param_opti.tasks.fusion import fusion_first_value_task, fusion_union_task
from pathlib import Path

# Using ground truth

# 1. execute PARIS pipeline, with different thresholds
# 2. evaluate the quality of the pipeline, with different thresholds


# - [ ] impl paris wrapper with exchange and threshold filter

seed_path = Path("data/seed.nt")
pipe_result_dir_path = Path("data/pipe_result")

def get_paris_pipeline(threshold: float):
    name = f"paris_graph_alignment_task={threshold}_fusion_first_value_task"

    return KgPipe(
        name=name,
        tasks=[paris_graph_alignment_task, fusion_first_value_task],
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=pipe_result_dir_path / "tmp"
    )

def test_paris_pipelines():
    pass