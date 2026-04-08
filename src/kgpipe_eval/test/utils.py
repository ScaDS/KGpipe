from pathlib import Path
from kgpipe.common import KG
from kgpipe.common.model.data import DataFormat
from kgpipe_eval.test.examples import *
from kgpipe_eval.api import MetricResult
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

def render_metric_result(metric_result: MetricResult) -> str:
    def _metric_key(mr: MetricResult) -> str:
        metric = mr.metric
        return getattr(metric, "key", metric.__class__.__name__)

    def _fmt_value(v) -> str:
        if isinstance(v, float):
            # stable, compact representation for test output
            return f"{v:.6g}"
        if isinstance(v, (int, bool)) or v is None:
            return str(v)
        if isinstance(v, str):
            return v
        if isinstance(v, Mapping):
            return json.dumps(v, indent=2, sort_keys=True, default=str)
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            return json.dumps(v, indent=2, sort_keys=True, default=str)
        return str(v)

    key = _metric_key(metric_result)
    summary = metric_result.summary or ""

    ms = sorted(metric_result.measurements, key=lambda m: m.name)
    name_w = max([len("measurement"), *(len(m.name) for m in ms)] or [len("measurement")])
    unit_w = max([len("unit"), *(len(m.unit or "") for m in ms)] or [len("unit")])

    lines: list[str] = []
    lines.append(f"metric: {key}")
    if summary:
        lines.append(f"summary: {summary}")
    if not ms:
        lines.append("(no measurements)")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"{'measurement':<{name_w}}  {'value'}{' ' * max(1, 2)}{'unit':<{unit_w}}")
    lines.append(f"{'-' * name_w}  {'-' * 20}  {'-' * unit_w}")

    for m in ms:
        unit = m.unit or ""
        rendered = _fmt_value(m.value)
        rendered_lines = rendered.splitlines() or [""]
        lines.append(f"{m.name:<{name_w}}  {rendered_lines[0]:<20}  {unit:<{unit_w}}")
        for cont in rendered_lines[1:]:
            lines.append(f"{'':<{name_w}}  {cont}")

    return "\n".join(lines)