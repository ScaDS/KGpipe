from kgpipe.common import TaskInput, TaskOutput
from kgpipe_examples.config import ExtendedFormats
from kgpipe.common.registry import Registry

@Registry.task(
    input_spec={"input": ExtendedFormats.SPECIAL_IN},
    output_spec={"output": ExtendedFormats.SPECIAL1}
)
def pipe_task_python(inputs: TaskInput, outputs: TaskOutput):
    # touch output file
    outputs["output"].path.touch()
    

@Registry.task(
    input_spec={"input": ExtendedFormats.SPECIAL1},
    output_spec={"output": ExtendedFormats.SPECIAL2}
)
def pipe_task_docker(inputs: TaskInput, outputs: TaskOutput):
    # touch output file
    outputs["output"].path.touch()

@Registry.task(
    input_spec={"input": ExtendedFormats.SPECIAL2},
    output_spec={"output": ExtendedFormats.SPECIAL_KG}
)
def pipe_task_remote(inputs: TaskInput, outputs: TaskOutput):
    # touch output file
    outputs["output"].path.touch()


