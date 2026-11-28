"""Dummy tasks for the example pipeline catalog.

The ports mirror the cases auto-connect has to resolve: a ``kg`` port beside
another N-Triples input, and two inputs of the same format.
"""

from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType
from kgpipe.common.model.data import DataFormat
from kgpipe.common.models import TaskInput, TaskOutput
from kgpipe.common.registry import Registry

THRESHOLD = ConfigurationDefinition(
    name="dummy-threshold",
    parameters=[
        Parameter(
            name="entity matching threshold",
            datatype=ParameterType.number,
            default_value=0.9,
            native_keys=["ENTITY_MATCHING_THRESHOLD"],
        ),
    ],
)


def _touch(outputs: TaskOutput) -> None:
    for data in outputs.values():
        data.path.parent.mkdir(parents=True, exist_ok=True)
        data.path.touch()


@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="Match source RDF against the seed KG",
    config_spec=THRESHOLD,
)
def dummy_match(inputs: TaskInput, outputs: TaskOutput):
    _touch(outputs)


@Registry.task(
    input_spec={"input": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.ER_JSON},
    description="Pass a match file through unchanged",
)
def dummy_exchange(inputs: TaskInput, outputs: TaskOutput):
    _touch(outputs)


@Registry.task(
    input_spec={"json1": DataFormat.ER_JSON, "json2": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.ER_JSON},
    description="Combine two match files",
)
def dummy_aggregate(inputs: TaskInput, outputs: TaskOutput):
    _touch(outputs)


@Registry.task(
    input_spec={
        "source": DataFormat.RDF_NTRIPLES,
        "kg": DataFormat.RDF_NTRIPLES,
        "matches1": DataFormat.ER_JSON,
    },
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Fuse source RDF into the seed using one match file",
    config_spec=THRESHOLD,
)
def dummy_fuse(inputs: TaskInput, outputs: TaskOutput):
    _touch(outputs)


@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES},
    output_spec={"result": DataFormat.RDF_NTRIPLES},
    description="Final N-Triples step",
)
def dummy_type(inputs: TaskInput, outputs: TaskOutput):
    _touch(outputs)
