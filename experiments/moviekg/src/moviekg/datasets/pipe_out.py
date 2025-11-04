from multiprocessing import Pipe
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

from kgpipe.common.models import KgPipePlan, KgStageReport

# benchmark_name/
# ├── results
# │   ├── pipeline_name_1
# │   │   ├── plan.json
# │   │   ├── stage_1
# │   │   │   ├── eval-report.json
# │   │   │   ├── exec-plan.json
# │   │   │   ├── exec-report.json
# │   │   │   └── tmp
# │   │   │       └── task_name_1_output.format
# │   │   └── stage_2
# │   │       └── ...
# │   └── pipeline_name_2
# │       └── ...
# ├── sources
# └── testing

class TaskOut(BaseModel):
    """
    TaskOut is a class that represents the output of a task.
    """
    task_name: str
    output: List[Path]

class StageOut(BaseModel):
    """
    StageOut is a class that represents the output of a stage.
    """
    root: Path
    stage_name: str
    tasks: List[TaskOut]
    resultKG: Optional[Path] = None
    plan: Optional[KgPipePlan] = None
    report: KgStageReport

class PipeOut(BaseModel):
    """
    PipeOut is a class that represents the output of a pipe.
    """
    root: Path
    pipeline_name: str
    stages: List[StageOut]
    resultKG: Optional[Path] = None


def load_pipe_out(path: Path) -> PipeOut:
    """
    Load a PipeOut object from a file.
    """

    pipe_out_root = path
    pipe_out_pipeline_name = path.name

    pipe_out_stages = []
    for stage_path in sorted(path.glob("stage_*")):
        stage_name = stage_path.name
        stage_plan = KgPipePlan.model_validate_json(open(stage_path / "exec-plan.json").read())
        stage_tasks = []
        for step in stage_plan.steps:
            stage_tasks.append(TaskOut(
                task_name=step.task,
                output=[stage_path / f"{output.path}" for output in step.output]
            ))
        
        stage_task_reports = KgStageReport.model_validate_json(open(stage_path / "exec-report.json").read())

        pipe_out_stages.append(StageOut(
            root=stage_path,
            stage_name=stage_name,
            tasks=stage_tasks,
            resultKG=stage_path / "result.nt",
            plan=stage_plan,
            report=stage_task_reports
        ))

    pipe_out = PipeOut(
        root=pipe_out_root, 
        pipeline_name=pipe_out_pipeline_name,
        stages=pipe_out_stages,
        resultKG=pipe_out_root / "result.nt",
    )
    return pipe_out