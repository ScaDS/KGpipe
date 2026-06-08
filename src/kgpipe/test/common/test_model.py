import json
from enum import Enum
from pathlib import Path

import pytest

from kgpipe.common.models import (
    BasicDataFormats,
    CustomDataFormats,
    Data,
    DataFormat,
    KgPipePlan,
    KgPipePlanStep,
)

class ProjectFormats(CustomDataFormats):
    EMBEDDINGS_JSON = "embeddings.json"


class ForeignFormats(str, Enum):
    MY_RAW = "my.raw"


def test_kg_pipe_plan_roundtrip():
    plan = KgPipePlan(
        steps=[
            KgPipePlanStep(
                task="paris_entity_matching",
                input=[Data(path=Path("data.nt"), format=DataFormat.RDF_NTRIPLES)],
                output=[Data(path=Path("data.paris_csv"), format=DataFormat.PARIS_CSV)],
            ),
            KgPipePlanStep(
                task="paris_csv_to_matching_format",
                input=[Data(path=Path("data.paris_csv"), format=DataFormat.PARIS_CSV)],
                output=[Data(path=Path("data.em_json"), format=DataFormat.ER_JSON)],
            ),
        ],
        seed=Data(path=Path("seed.nt"), format=DataFormat.RDF_NTRIPLES),
        source=Data(path=Path("source.nt"), format=DataFormat.RDF_NTRIPLES),
        result=Data(path=Path("result.nt"), format=DataFormat.RDF_NTRIPLES),
    )

    plan_json = plan.model_dump_json()
    plan_back = KgPipePlan(**json.loads(plan_json))

    assert plan == plan_back


def test_data_accepts_basic_data_formats():
    data = Data(path=Path("a.nt"), format=BasicDataFormats.RDF_NTRIPLES)
    assert data.format == BasicDataFormats.RDF_NTRIPLES
    assert data.to_dict()["format"] == "nt"


def test_data_accepts_custom_data_formats():
    data = Data(path=Path("embed.json"), format=ProjectFormats.EMBEDDINGS_JSON)
    assert data.format == ProjectFormats.EMBEDDINGS_JSON
    assert data.to_dict()["format"] == "embeddings.json"


def test_data_rejects_foreign_string_enum_not_based_on_custom_catalog():
    with pytest.raises(ValueError):
        Data(path=Path("x.raw"), format=ForeignFormats.MY_RAW)


def test_data_rejects_unknown_string_format():
    with pytest.raises(ValueError, match="Unknown format: does-not-exist"):
        Data(path=Path("x.any"), format="does-not-exist")