from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from kgpipe.common.models import KgPipePlan, KgStageReport


class TaskOut(BaseModel):
    """
    Output artifacts produced by a single task within a stage.
    """

    task_name: str
    output: List[Path]


class StageOut(BaseModel):
    """
    Output artifacts for one incremental stage.
    """

    root: Path
    stage_name: str
    tasks: List[TaskOut]
    resultKG: Optional[Path] = None
    plan: Optional[KgPipePlan] = None
    report: KgStageReport

    @property
    def stage_index(self) -> int:
        """
        Extract stage number from `stage_<n>` directory name.
        """
        return int(self.stage_name.split("_", 1)[1])


class PipeOut(BaseModel):
    """
    Output artifacts for a full incremental pipeline run directory containing stage_* subdirs.
    """

    root: Path
    pipeline_name: str
    stages: List[StageOut]
    resultKG: Optional[Path] = None


def _stage_paths(run_dir: Path) -> list[Path]:
    stage_paths = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("stage_")]
    stage_paths.sort(key=lambda p: int(p.name.split("_", 1)[1]))
    return stage_paths


def _resolve_stage_result_kg(stage_dir: Path) -> Path:
    """
    Prefer `result_eval.nt` (evaluation-ready), fallback to `result.nt`.
    """
    candidates = [
        # stage_dir / "result_eval.nt",
        stage_dir / "result.nt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Keep the legacy default for downstream tools that expect result.nt even if not created yet.
    return stage_dir / "result.nt"


def load_stage_out(stage_dir: Path) -> StageOut:
    """
    Load stage outputs from a `stage_<n>` directory produced by KGpipe incremental runs.
    """
    stage_name = stage_dir.name

    plan_path = stage_dir / "exec-plan.json"
    report_path = stage_dir / "exec-report.json"

    if not plan_path.exists():
        raise FileNotFoundError(f"Missing exec plan: {plan_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing exec report: {report_path}")

    stage_plan = KgPipePlan.model_validate_json(plan_path.read_text())

    stage_tasks: list[TaskOut] = []
    for step in stage_plan.steps:
        stage_tasks.append(
            TaskOut(
                task_name=step.task,
                output=[stage_dir / f"{output.path}" for output in step.output],
            )
        )

    stage_report = KgStageReport.model_validate_json(report_path.read_text())

    return StageOut(
        root=stage_dir,
        stage_name=stage_name,
        tasks=stage_tasks,
        resultKG=_resolve_stage_result_kg(stage_dir),
        plan=stage_plan,
        report=stage_report,
    )


def load_pipe_out(run_dir: Path) -> PipeOut:
    """
    Load a pipeline run output directory that contains `stage_*` directories.
    """
    run_dir = Path(run_dir)
    stages = [load_stage_out(p) for p in _stage_paths(run_dir)]

    return PipeOut(
        root=run_dir,
        pipeline_name=run_dir.name,
        stages=stages,
        resultKG=_resolve_stage_result_kg(run_dir) if (run_dir / "result.nt").exists() else (run_dir / "result.nt"),
    )

