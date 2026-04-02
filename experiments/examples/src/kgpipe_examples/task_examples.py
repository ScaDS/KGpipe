from kgpipe.common import TaskInput, TaskOutput
from kgpipe.common import trace_task_run
from kgpipe.common.model.configuration import ConfigurationProfile, ConfigurationDefinition, Parameter, ParameterType
from kgpipe_examples.config import ExtendedFormats
from kgpipe.common.model.default_catalog import BasicTaskCategoryCatalog
from kgpipe.common.registry import Registry

@trace_task_run
@Registry.task(
    input_spec={"input": ExtendedFormats.SPECIAL_IN},
    output_spec={"output": ExtendedFormats.SPECIAL1},
    category=[BasicTaskCategoryCatalog.entity_resolution],
    description="A task that processes a special input and produces a special output"
)
def pipe_task_python(inputs: TaskInput, outputs: TaskOutput):
    # touch output file
    outputs["output"].path.touch()
    

# def converts_pdfs: pass
# def extracts_text

@trace_task_run
@Registry.task(
    input_spec={"input": ExtendedFormats.SPECIAL1},
    output_spec={"output": ExtendedFormats.SPECIAL2}
)
def pipe_task_docker(inputs: TaskInput, outputs: TaskOutput):
    # touch output file
    outputs["output"].path.touch()

@trace_task_run
@Registry.task(
    input_spec={"input": ExtendedFormats.SPECIAL2},
    output_spec={"output": ExtendedFormats.SPECIAL_KG}
)
def pipe_task_remote(inputs: TaskInput, outputs: TaskOutput):
    # touch output file
    outputs["output"].path.touch()


@trace_task_run
@Registry.task(
    input_spec={"input": ExtendedFormats.SPECIAL1},
    output_spec={"output": ExtendedFormats.SPECIAL_KG},
    category=[BasicTaskCategoryCatalog.entity_resolution],
    config_spec=ConfigurationDefinition(
        name="pipe_task_with_config_spec",
        description="Configuration specification for the pipe_task_with_config task",
        parameters=[
            Parameter(
                name="some_parameter",
                native_keys=["some_parameter"],
                datatype=ParameterType.string,
                default_value="default",
                required=False,
                allowed_values=[]
            )
        ]
    )
)
def pipe_task_with_config(inputs: TaskInput, outputs: TaskOutput, config: ConfigurationProfile):
    # print config
    print(config)
    outputs["output"].path.touch()