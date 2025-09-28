from kgpipe.common.models import DataFormat, KgTask, Data
from kgpipe.common.registry import Registry
from pathlib import Path

def dummy_function(input_data: dict[str, Data], output_data: dict[str, Data]):
    pass

# @Registry.register("task")
# def dummy_task() -> KgTask:
#     return KgTask(
#         name="dummy_task",
#         input_spec={"in": DataFormat.ANY},
#         output_spec={"out": DataFormat.ANY},
#         function=dummy_function
#     )

@Registry.task({"in": DataFormat.ANY}, {"out": DataFormat.ANY})
def dummy_kg_task(i: dict[str, Data], o: dict[str, Data]):
    print(f"Processing {i['in'].path} to {o['out'].path}")


def test_registry_task():
    task_1: KgTask = Registry.get("task","dummy_kg_task")
    report = task_1.run(
        inputs=[Data(path=Path("in"), format=DataFormat.ANY)], 
        outputs=[Data(path=Path("out"), format=DataFormat.ANY)]
        )
    assert report.status == "success"

    task_2 = Registry.get_task("dummy_kg_task")
    report = task_2.run(
        inputs=[Data(path=Path("in"), format=DataFormat.ANY)], 
        outputs=[Data(path=Path("out"), format=DataFormat.ANY)]
        )
    assert report.status == "success"



def test_registry_list():
    tasks = Registry.list("task")
    for task in tasks:
        print(f"{task.name} input: {task.input_spec} output: {task.output_spec}")


# def test_registry():
#     task: KgTask = Registry.get("task","dummy_task").__call__()
#     print(task.name)