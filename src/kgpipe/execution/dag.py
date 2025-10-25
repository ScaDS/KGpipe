# """
# DAG-based execution support for pipelines.
# """

# from __future__ import annotations

# import asyncio
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional, Set
# from collections import defaultdict, deque

# from pathlib import Path
# from ..common.models import Data, DataFormat, KgPipe, KgTaskReport, Stage, TaskStatus
# from .runner import ExecutionReport, PipelineRunner


# @dataclass
# class DagNode:
#     """A node in the execution DAG."""
    
#     name: str
#     stage: Optional[Stage] = None
#     task: Optional[Any] = None  # KgTask
#     dependencies: Set[str] = field(default_factory=set)
#     dependents: Set[str] = field(default_factory=set)
#     status: TaskStatus = TaskStatus.PENDING
#     result: Optional[ExecutionReport] = None


# class DagExecutor:
#     """Executor for DAG-based pipeline execution."""
    
#     def __init__(self, runner: Optional[PipelineRunner] = None):
#         self.runner = runner or PipelineRunner()
#         self.nodes: Dict[str, DagNode] = {}
#         self.execution_order: List[str] = []
    
#     def build_dag_from_pipeline(self, pipeline: KgPipe) -> None:
#         """Build a DAG from a pipeline."""
#         self.nodes.clear()
#         self.execution_order.clear()
        
#         # Create nodes for each task
#         for task in pipeline.tasks:
#             node = DagNode(name=task.name, task=task)
#             self.nodes[task.name] = node
        
#         # Build dependencies based on data flow
#         # This is a simplified version - in practice, you'd analyze input/output specs
#         task_names = [task.name for task in pipeline.tasks]
#         for i, task in enumerate(pipeline.tasks):
#             # For now, assume sequential dependencies
#             if i > 0:
#                 prev_task = task_names[i - 1]
#                 self.nodes[task.name].dependencies.add(prev_task)
#                 self.nodes[prev_task].dependents.add(task.name)
        
#         # Calculate execution order using topological sort
#         self.execution_order = self._topological_sort()
    
#     def build_dag_from_stages(self, stages: List[Stage]) -> None:
#         """Build a DAG from a list of stages."""
#         self.nodes.clear()
#         self.execution_order.clear()
        
#         # Create nodes for each stage
#         for stage in stages:
#             node = DagNode(name=stage.name, stage=stage)
#             self.nodes[stage.name] = node
        
#         # Build dependencies from stage.depends_on
#         for stage in stages:
#             node = self.nodes[stage.name]
#             for dep in stage.depends_on:
#                 if dep in self.nodes:
#                     node.dependencies.add(dep)
#                     self.nodes[dep].dependents.add(stage.name)
        
#         # Calculate execution order using topological sort
#         self.execution_order = self._topological_sort()
    
#     def _topological_sort(self) -> List[str]:
#         """Perform topological sort to determine execution order."""
#         # Calculate in-degrees
#         in_degree = {name: len(node.dependencies) for name, node in self.nodes.items()}
        
#         # Find nodes with no dependencies
#         queue = deque([name for name, degree in in_degree.items() if degree == 0])
#         result = []
        
#         while queue:
#             node_name = queue.popleft()
#             result.append(node_name)
            
#             # Reduce in-degree for dependents
#             for dependent in self.nodes[node_name].dependents:
#                 in_degree[dependent] -= 1
#                 if in_degree[dependent] == 0:
#                     queue.append(dependent)
        
#         # Check for cycles
#         if len(result) != len(self.nodes):
#             raise ValueError("Pipeline has circular dependencies")
        
#         return result
    
#     def execute_sequential(self, input_data: List[Data]) -> ExecutionReport:
#         """Execute the DAG sequentially."""
#         start_time = asyncio.get_event_loop().time()
        
#         try:
#             task_reports = []
            
#             for node_name in self.execution_order:
#                 node = self.nodes[node_name]
#                 node.status = TaskStatus.RUNNING
                
#                 if node.task:
#                     # Execute task
#                     # This is simplified - you'd need to properly map inputs/outputs
#                     mock_inputs = input_data[:1] if input_data else []
#                     mock_outputs = [Data(Path(f"output_{node_name}.ttl"), DataFormat.TTL)]
                    
#                     report = node.task.run(mock_inputs, mock_outputs)
#                     task_reports.append(report)
                    
#                     if report.status == "failed":
#                         node.status = TaskStatus.FAILED
#                         raise RuntimeError(f"Task {node_name} failed")
#                     else:
#                         node.status = TaskStatus.COMPLETED
                
#                 elif node.stage:
#                     # Execute stage
#                     report = self.runner.stage_runner.run_stage(node.stage, input_data)
#                     task_reports.extend(report.task_reports)
                    
#                     if not report.success:
#                         node.status = TaskStatus.FAILED
#                         raise RuntimeError(f"Stage {node_name} failed")
#                     else:
#                         node.status = TaskStatus.COMPLETED
            
#             end_time = asyncio.get_event_loop().time()
            
#             return ExecutionReport(
#                 pipeline_name="dag_pipeline",
#                 start_time=start_time,
#                 end_time=end_time,
#                 status=TaskStatus.COMPLETED,
#                 task_reports=task_reports
#             )
            
#         except Exception as e:
#             end_time = asyncio.get_event_loop().time()
            
#             return ExecutionReport(
#                 pipeline_name="dag_pipeline",
#                 start_time=start_time,
#                 end_time=end_time,
#                 status=TaskStatus.FAILED,
#                 error_message=str(e)
#             )
    
#     async def execute_parallel(self, input_data: List[Data], max_concurrent: int = 4) -> ExecutionReport:
#         """Execute the DAG with parallel execution where possible."""
#         start_time = asyncio.get_event_loop().time()
        
#         try:
#             task_reports = []
#             semaphore = asyncio.Semaphore(max_concurrent)
            
#             # Create tasks for all nodes
#             tasks = []
#             for node_name in self.execution_order:
#                 task = asyncio.create_task(
#                     self._execute_node_with_dependencies(node_name, input_data, semaphore)
#                 )
#                 tasks.append(task)
            
#             # Wait for all tasks to complete
#             results = await asyncio.gather(*tasks, return_exceptions=True)
            
#             # Collect results
#             for result in results:
#                 if isinstance(result, Exception):
#                     raise result
#                 if isinstance(result, list):
#                     task_reports.extend(result)
            
#             end_time = asyncio.get_event_loop().time()
            
#             return ExecutionReport(
#                 pipeline_name="dag_pipeline",
#                 start_time=start_time,
#                 end_time=end_time,
#                 status=TaskStatus.COMPLETED,
#                 task_reports=task_reports
#             )
            
#         except Exception as e:
#             end_time = asyncio.get_event_loop().time()
            
#             return ExecutionReport(
#                 pipeline_name="dag_pipeline",
#                 start_time=start_time,
#                 end_time=end_time,
#                 status=TaskStatus.FAILED,
#                 error_message=str(e)
#             )
    
#     async def _execute_node_with_dependencies(self, node_name: str, input_data: List[Data], 
#                                             semaphore: asyncio.Semaphore) -> List[KgTaskReport]:
#         """Execute a single node, waiting for its dependencies."""
#         node = self.nodes[node_name]
        
#         # Wait for dependencies to complete
#         for dep_name in node.dependencies:
#             dep_node = self.nodes[dep_name]
#             while dep_node.status != TaskStatus.COMPLETED:
#                 if dep_node.status == TaskStatus.FAILED:
#                     raise RuntimeError(f"Dependency {dep_name} failed")
#                 await asyncio.sleep(0.1)
        
#         # Execute the node
#         async with semaphore:
#             node.status = TaskStatus.RUNNING
            
#             if node.task:
#                 # Execute task
#                 mock_inputs = input_data[:1] if input_data else []
#                 mock_outputs = [Data(Path(f"output_{node_name}.ttl"), DataFormat.TTL)]
                
#                 report = node.task.run(mock_inputs, mock_outputs)
                
#                 if report.status == "failed":
#                     node.status = TaskStatus.FAILED
#                     raise RuntimeError(f"Task {node_name} failed")
#                 else:
#                     node.status = TaskStatus.COMPLETED
#                     return [report]
            
#             elif node.stage:
#                 # Execute stage
#                 report = self.runner.stage_runner.run_stage(node.stage, input_data)
                
#                 if not report.success:
#                     node.status = TaskStatus.FAILED
#                     raise RuntimeError(f"Stage {node_name} failed")
#                 else:
#                     node.status = TaskStatus.COMPLETED
#                     return report.task_reports
            
#             return []
    
#     def get_execution_plan(self) -> Dict[str, Any]:
#         """Get the execution plan as a dictionary."""
#         return {
#             "execution_order": self.execution_order,
#             "nodes": {
#                 name: {
#                     "dependencies": list(node.dependencies),
#                     "dependents": list(node.dependents),
#                     "status": node.status.value
#                 }
#                 for name, node in self.nodes.items()
#             }
#         } 