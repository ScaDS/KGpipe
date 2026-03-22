"""
KGflex model.py
Core domain model for the KGflex framework.

This module defines the core data structures that represent the domain model
for generating and executing KG pipelines.
"""

from __future__ import annotations

from .model.data import Data, DataFormat, DynamicFormat, DataSet, FormatRegistry
from .model.task import KgTask, KgTaskReport
from .model.pipeline import KgPipe, KgPipePlan, KgPipePlanStep, KgStageReport
from .model.evaluation import Metric, EvaluationReport
from .model.kg import KG
from .model.task import TaskInput, TaskOutput

__all__ = [
    "Data", "DataFormat", "DynamicFormat", "DataSet", "FormatRegistry", "KgTask", "KgTaskReport", "KgPipe", "KgPipePlan", "KgPipePlanStep", "KgStageReport",  "Metric", "EvaluationReport", "KG", "TaskInput", "TaskOutput"
]
