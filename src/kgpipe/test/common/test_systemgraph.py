from kgpipe.common.systemgraph import kg_class, kg_function, SYS_KG, add_task, add_task_result, add_pipeline, add_pipeline_result
from kgpipe.common.definitions import Task, Eval, Pipeline, TaskResult, DataHandle, PipelineResult
import sys
from kgcore.backend.rdf.rdf_rdflib import RDFLibBackend

task1 = Task(
    name="test_task", 
    type="test_type", 
    description="test_description", 
    input=["test_input"], 
    output=["test_output"]
)
task2 = Task(
    name="test_task2", 
    type="test_type2", 
    description="test_description2", 
    input=["test_input2"], 
    output=["test_output2"]
)
task_result1 = TaskResult(
    task=task1, 
    config={"test_config": "test_config"}, 
    input=[DataHandle(uri="test_input", type="test_input_type")], 
    output=[DataHandle(uri="test_output", type="test_output_type")],
    status="test_status",
    duration=10.0
)
task_result2 = TaskResult(
    task=task2, 
    config={"test_config2": "test_config2"}, 
    input=[DataHandle(uri="test_input2", type="test_input_type2")], 
    output=[DataHandle(uri="test_output2", type="test_output_type2")],
    status="test_status2",
    duration=20.0
)
pipeline = Pipeline(
    tasks=[task1, task2], 
    input=["test_input"], 
    output=["test_output"]
)
pipeline_result = PipelineResult(
    task_results=[task_result1, task_result2],
    eval_results=[],
    input=[DataHandle(uri="test_input", type="test_input_type")],
    output=[DataHandle(uri="test_output", type="test_output_type")],
    status="test_status",
    duration=30.0
)

model: RDFLibBackend = SYS_KG.backend

def test_task_entity():
    add_task(task1)
    add_task(task2)

    # print(model.get_rdflibgraph().serialize(format="turtle"))

def test_task_result_entity():
    add_task_result(task_result1)
    add_task_result(task_result2)

    # print(model.get_rdflibgraph().serialize(format="turtle"))

def test_pipeline_entity():
    add_pipeline(pipeline)

    # print(model.get_rdflibgraph().serialize(format="turtle"))

def test_pipeline_result_entity():
    add_pipeline_result(pipeline_result)
    
    print(model.get_rdflibgraph().serialize(format="turtle"))