# KG Evaluation (new API)

KGpipe currently contains **two** evaluation implementations:

- **New** (recommended): `kgpipe_eval` (package: `src/kgpipe_eval/`)
- **Old** (deprecated soon): `kgpipe.evaluation` (package: `src/kgpipe/evaluation/`)

This page documents the **new** `kgpipe_eval` API.

## Mental model

In `kgpipe_eval`, evaluation is composed from:
- **KG loader / adapter**: turns a `KgLike` (e.g. a `kgpipe.common.model.kg.KG`) into an in-memory `TripleGraph`
- **Metric instances**: objects implementing `Metric.compute(...) -> MetricResult`
- **Metric configs** (optional): typed config objects passed to metrics that require parameters
- **Evaluator**: runs multiple metrics against a loaded graph

Key types:
- `kgpipe_eval.api.Metric`: metric interface (`key`, `description`, `compute`)
- `kgpipe_eval.api.MetricResult`: dataclass with `measurements` + optional `summary`
- `kgpipe_eval.evaluator.Evaluator`: runs a list of metrics with an optional `confs` dict

## Minimal example (statistics)

```python
from pathlib import Path

from kgpipe.common.model.data import DataFormat
from kgpipe.common.model.kg import KG

from kgpipe_eval.evaluator import Evaluator
from kgpipe_eval.metrics.statistics import CountMetric
from kgpipe_eval.utils.kg_utils import KgManager

kg = KG(
    id="my_kg",
    name="My KG",
    path=Path("my_kg.nt"),
    format=DataFormat.RDF_NTRIPLES,
)

tg = KgManager.load_kg(kg)
results = Evaluator().run(tg, metrics=[CountMetric()])

for r in results:
    print(r.metric.key, r.summary)
    for m in r.measurements:
        print(" ", m.name, m.value)
```

## Metrics that need configuration

Some metrics require a config object. The `Evaluator` detects this by introspecting the metric’s
`compute(...)` signature:
- `compute(self, kg)` → no config needed
- `compute(self, kg, config)` → config required and must be provided

You pass configs via a dict keyed by the metric key/class name.

Example (triple alignment + duplicates):

```python
from kgpipe_eval.evaluator import Evaluator
from kgpipe_eval.utils.kg_utils import KgManager

from kgpipe_eval.metrics.duplicates import DuplicateMetric, DuplicateConfig
from kgpipe_eval.metrics.triple_alignment import TripleAlignmentMetric, TripleAlignmentConfig
from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig

tg = KgManager.load_kg("path/to/result_eval.nt")  # KgLike: path, KG object, ...

metrics = [DuplicateMetric(), TripleAlignmentMetric()]
confs = {
    "DuplicateMetric": DuplicateConfig(
        entity_alignment_config=EntityAlignmentConfig(
            method="label_embedding",
            verified_entities_path="path/to/verified_entities.tsv",
            verified_entities_delimiter="\\t",
            entity_sim_threshold=0.95,
        )
    ),
    "TripleAlignmentMetric": TripleAlignmentConfig(
        reference_kg="path/to/reference.nt",
        entity_alignment_config=EntityAlignmentConfig(
            method="label_embedding",
            reference_kg="path/to/reference.nt",
            entity_sim_threshold=0.95,
        ),
        value_sim_threshold=0.5,
        cache_literal_embeddings=True,
        cache_ref_literal_embeddings=True,
    ),
}

results = Evaluator().run(tg, metrics, confs)
```

## Canonical reference example (MovieKG)

For a realistic end-to-end usage example (loading pipeline stage outputs, wiring configs, running multiple metrics),
see:

- `experiments/moviekg/src/moviekg/evaluation/test_eval_refactor.py`

That file shows how to:
- build per-metric configs (duplicates/entity alignment/triple alignment)
- load the KG from a pipeline output directory
- serialize `MetricResult` to JSON (because it contains metric objects)

## CLI note

There is a “new eval” CLI command path intended to run these metrics (see `kgpipe_eval.api` docstring mentioning
`kgpipe eval-new`). If you want the docs to include the CLI, we should first confirm the exact CLI flags and expected
inputs in `src/kgpipe/cli/eval_new.py` and align this page with that implementation.