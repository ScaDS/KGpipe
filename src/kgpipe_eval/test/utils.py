from pathlib import Path
from kgpipe.common import KG
from kgpipe.common.model.data import DataFormat
from kgpipe_eval.test.examples import *
from kgpipe_eval.api import MetricResult
from kgpipe_eval.utils.metric_utils import render_metric_result
from rdflib import Graph
import json
from collections.abc import Mapping, Sequence

tmp_dir = Path("tmp_test_data")

if not tmp_dir.exists():
    tmp_dir.mkdir(parents=True, exist_ok=True)


def get_test_kg(sample_size: int = -1) -> KG:
    test_triples = TEST_TURTLE_TRIPLES
    if sample_size > 0:
        test_triples = test_triples[:sample_size]
    # write test_triples to a file
    g = Graph()
    g.parse(data=test_triples, format="turtle")
    g.serialize(destination=tmp_dir / "test.nt", format="ntriples")
    return KG("test", name="test", path=tmp_dir / "test.nt", format=DataFormat.RDF_NTRIPLES)

def get_reference_kg(sample_size: int = -1) -> KG:
    reference_triples = REFERENCE_TURTLE_TRIPLES
    if sample_size > 0:
        reference_triples = reference_triples[:sample_size]
    # write reference_triples to a file
    g = Graph()
    g.parse(data=reference_triples, format="turtle")
    g.serialize(destination=tmp_dir / "reference.nt", format="ntriples")
    return KG("reference", name="reference", path=tmp_dir / "reference.nt", format=DataFormat.RDF_NTRIPLES)

def get_verified_entities_path() -> Path:
    path = tmp_dir / "verified_entities.csv"
    with open(path, "w") as f:
        # Avoid a leading blank line which breaks csv.DictReader header parsing
        f.write(VERIFIED_ENTITIES.lstrip().replace("o:", "http://example.org/ontology/"))
    return path