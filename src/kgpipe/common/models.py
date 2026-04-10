"""
KGflex model.py
Core domain model for the KGflex framework.

This module defines the core data structures that represent the domain model
for generating and executing KG pipelines.
"""

from __future__ import annotations

from .model.data import Data, DataFormat, DataSet
from .model.default_catalog import BasicDataFormats, CustomDataFormats, BasicTaskCategoryCatalog
from .model.task import KgTask, KgTaskReport
from .model.pipeline import KgPipe, KgPipePlan, KgPipePlanStep, KgStageReport
from .model.evaluation import Metric, EvaluationReport
from .model.kg import KG
from .model.task import TaskInput, TaskOutput, KgTask, KgTaskRun
# from .model.evaluation import KgMetric, KgMetricRun

__all__ = [
    "Data", "DataFormat", "BasicDataFormats", "CustomDataFormats", "BasicTaskCategoryCatalog", "DataSet", "KgTask", "KgTaskReport", "KgPipe", "KgPipePlan", "KgPipePlanStep", "KgStageReport",  "Metric", "EvaluationReport", "KG", "TaskInput", "TaskOutput", "KgTaskRun"
]
