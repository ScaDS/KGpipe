from kgpipe.common import TaskInput, TaskOutput
from kgpipe.common.model.configuration import ConfigurationProfile, ConfigurationDefinition, Parameter, ParameterType
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


@Registry.task(
    input_spec={"input": ExtendedFormats.SPECIAL2},
    output_spec={"output": ExtendedFormats.SPECIAL_KG},
    config_spec=ConfigurationDefinition(
        name="pipe_task_with_config_spec",
        description="Configuration specification for the pipe_task_with_config task",
        parameters=[
            Parameter(name="some_parameter", datatype=ParameterType.string, default_value="default", required=False)
        ]
    )
)
def pipe_task_with_config(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    # print config
    print(config)
    outputs["output"].path.touch()