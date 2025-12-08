# tests/test_smoke.py
import tempfile
import os
from kgcore.api import KGGraph, KG
import pytest

def roundtrip_test(kg: KGGraph):
    a = kg.create_entity(["A"], {"x": 1})
    b = kg.create_entity(["B"], {"y": 2})
    r = kg.create_relation("R", a.id, b.id, {"w": 3})
    kg.add_meta(a.id, {"m": 9})
    assert kg.find_entities("A")[0].props["x"] == 1
    assert kg.find_relations("R")[0].props["w"] == 3
    print(kg.find_entities(None,None))

def test_memory_roundtrip():
    kg: KGGraph = KG()
    roundtrip_test(kg)

@pytest.mark.skip(reason="RDF file backend is not implemented yet")
def test_rdf_file_roundtrip():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_kg.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        roundtrip_test(kg)