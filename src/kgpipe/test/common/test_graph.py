from uuid import uuid4

from kgpipe.common.graph.definitions import (
    DataTypeEntity,
    DataSpecEntity,
    TaskEntity,
    ImplementationEntity,
)
from kgpipe.common.graph.systemgraph import PipeKG


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def test_add_implementation_and_find_implemenetation():
    task = TaskEntity(name=_uid("task"), description="test task")
    task_id = PipeKG.add_task(task)

    data_type = DataTypeEntity(format="text/csv", data_schema=_uid("schema"))
    data_type_id = PipeKG.add_data_type(data_type)

    in_spec_id = PipeKG.add_data_spec(DataSpecEntity(name=_uid("in_spec"), data_type=data_type_id))
    out_spec_id = PipeKG.add_data_spec(DataSpecEntity(name=_uid("out_spec"), data_type=data_type_id))

    impl_name = _uid("impl")
    impl = ImplementationEntity(
        name=impl_name,
        version="0.0.1",
        input_spec=[in_spec_id],
        output_spec=[out_spec_id],
        realizesTask=[task_id],
        usesTool=[],
    )

    PipeKG.add_implementation(impl)
    found = PipeKG.find_implementation(impl_name)

    assert found is not None
    assert found.name == impl_name
    assert found.version == "0.0.1"
