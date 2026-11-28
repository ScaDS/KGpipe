# Pipeline specification and execution

A pipeline catalog is a user specification. `KgPipe.build()` resolves it into a `KgPipePlan`: concrete files, the port each file was bound to, the parameter values, and a fingerprint of that candidate. The catalog does not itself store the task DAG.

## Catalog

The catalog schema version is `kgpipe.pipeline/v1`, written as the root `schema` key. That key is not a pipeline. Files that omit it still load as v1. An unknown `schema` value is rejected. `experiments/examples/sample.yaml` is a single-pipeline file in this form. `experiments/examples/pipeline.conf` is a larger catalog of the same dummy tasks (`experiments/examples/src/kgpipe_examples/dummy_tasks.py`). `experiments/moviekg/pipeline.conf` is the MovieKG catalog: `rdf_base` uses typed `parameters` and port declarations; other entries still use the legacy `config` map and an ordered list of task names. `connect` defaults to `auto`.

```yaml
schema: kgpipe.pipeline/v1

rdf_base:
    description: "Align source RDF with target KG"
    connect: auto
    inputs:
        source: {role: source, format: nt}
        seed: {role: seed, format: nt}
    output:
        result: {role: result, format: nt}
    parameters:
        entity_matching_threshold: 0.99
        relation_matching_threshold: 0.5
    tasks:
        - paris_entity_matching
        - paris_exchange
        - fusion_first_value
        - type_inference_ontology_simple
```

`schema` is the catalog version, not a pipeline. Existing files omit it and still load. A task is either a registry name or a step:

```yaml
tasks:
    - name: paris_entity_matching
      parameters:
          entity_matching_threshold: 0.99
```

Load a catalog and run it without assembling `KgPipe` by hand:

```python
from kgpipe.generation.loaders import build_from_yaml, validate_pipeline

pipe = build_from_yaml("pipeline.yaml", "rdf_base")
reports = pipe.run()
```

`validate_pipeline(path, name)` is the same load and plan step without `run()`. Invalid task names, backends, parameter types, and port combinations fail there.

`build_from_conf` is the lower-level form when the files already exist as `Data` objects:

```python
from kgpipe.generation.loaders import load_pipeline_catalog, build_from_conf
from kgpipe.common.models import Data, DataFormat

catalog = load_pipeline_catalog(path)
conf = catalog.root["rdf_base"]
seed = Data(path="seed.nt", format=DataFormat.RDF_NTRIPLES)
pipe = build_from_conf("rdf_base", conf, seed, data_dir)
plan = pipe.build(source=source, result=result, stable_files=True)
```

`build_from_conf` resolves task names through the registry and binds parameters onto each task that declares `config_spec`. It does not connect ports. That happens in `build()`.

A step may set `id`, `task` (or `name`), `parameters`, `inputs`, `outputs`, `backend`, and `backend_options`. `backend` is `python`, `docker`, or `remote`. If the catalog asks for a backend the registered task does not have, load fails. `backend_options` is a free-form map for that backend; it is not part of the fingerprint.

## Defaults, paths, and environment

Catalog-level `defaults` and `env` are reserved keys, not pipelines. Pipeline `defaults` overlay the catalog. `build_from_yaml` then overlays function arguments. Relative paths are resolved against the YAML file directory. File paths never enter `plan.fingerprint`.

```yaml
schema: kgpipe.pipeline/v1

env:
    KGPIPE_THRESHOLD: "0.99"

defaults:
    data_dir: ./output
    source: ./data/source.nt
    seed: ${SEED_FILE:-./data/seed.nt}
    result: ./output/result.nt
    ontology: ./data/ontology.ttl
```

`seed` and `target` are the same slot. `${VAR}` and `${VAR:-default}` are expanded when the catalog loads. Bare `$source`, `$seed`, and `$result` are port refs and are not expanded. A missing variable without a default is an error. `load_pipeline_catalog(..., environ=...)` supplies the map; the default is `os.environ` overlaid on the catalog `env` block.

## Auto-connect

`connect: auto` keeps a catalog that starts as `[source, seed]`. For each task, in `input_spec` order:

- A port named `kg` is bound to the last catalog entry. Outputs are prepended, so that entry stays the seed.
- Every other port takes the first entry whose format matches and that this task has not already used.
- Outputs are new files in `data_dir`. The last task's output is `result` when `build()` was given one.
- Those outputs are prepended, so the next task sees the newest artifact first.

The resolved plan records a ref for every port: `$source`, `$seed`, `$result`, or `step_id.port`. For `rdf_base` that is:

| Step | Inputs | Output |
|---|---|---|
| `paris_entity_matching` | `source=$source`, `kg=$seed` | `output=paris_entity_matching.output` (`paris.csv`) |
| `paris_exchange` | `input=paris_entity_matching.output` | `output=paris_exchange.output` (`er.json`) |
| `fusion_first_value` | `source=$source`, `kg=$seed`, `matches1=paris_exchange.output` | `output=fusion_first_value.output` (`nt`) |
| `type_inference_ontology_simple` | `source=fusion_first_value.output` | `result=$result` |

`kg` is part of the algorithm. `fusion_first_value` has two N-Triples inputs, and format matching alone cannot tell source from seed. Two inputs of the same format are bound newest-first: `aggregate_2matches` assigns `json1` the newer `er.json` and `json2` the older one.

`kgpipe run pipeline.conf rdf_base --dry-run` prints this wiring and the candidate fingerprint.

## Explicit ports

The same refs can be written in the catalog. This is the form to use when auto-connect is ambiguous, or when the resolved plan is dumped back out. Bindings on a step require `connect: explicit`.

```yaml
rdf_base:
    description: "Align source RDF with target KG"
    connect: explicit
    tasks:
        - id: match
          task: paris_entity_matching
          inputs: {source: $source, kg: $seed}
          outputs: {output: match.output}
        - id: exchange
          task: paris_exchange
          inputs: {input: match.output}
          outputs: {output: exchange.output}
        - id: fuse
          task: fusion_first_value
          inputs: {source: $source, kg: $seed, matches1: exchange.output}
          outputs: {output: fuse.output}
        - id: types
          task: type_inference_ontology_simple
          inputs: {source: fuse.output}
          outputs: {result: $result}
```

`build()` checks that every port exists on the task, that each ref is `$source`, `$seed`, `$result`, or an earlier `step_id.port`, and that the formats agree. `$result` is only valid on the last step.

## Fingerprint

`plan.fingerprint` is a sha256 of the schema version, step id, task name, port refs, and resolved parameter values. Paths are not included, so the same candidate run in two directories has one fingerprint. Changing a parameter or a port ref changes it.

## Search candidates

A parameter-search candidate is the same catalog entry. The search still samples a list of live `KgTask` objects and a `ConfigurationProfile` per task. `build_candidate()` writes that into a `kgpipe.pipeline/v1` `PipelineConf` before `build()`. Task order is the task list. Parameter values are the profile bindings. `connect` stays `auto`.

```python
from kgpipe.generation.loaders import build_candidate

pipe, conf = build_candidate(name, tasks, profiles, seed, data_dir)
plan = pipe.build(source=source, result=result, stable_files=True, configCatalog=pipe.config_catalog)
```

`dump_pipeline_catalog()` writes that entry as YAML, and `load_pipeline_catalog()` reads it back. `pipeline_conf_from_plan()` writes the resolved port refs as `connect: explicit`. Rebuilding either form produces the same task order, port refs, parameter values, and fingerprint.

The sampled RDF runs in `data/kgpipe-experiment-results/parameter_search/rdf_sampled` keep the older `{task_keys, profiles}` JSON beside the resolved plan. Those plans record the files that were wired together. Rebuilding the candidate from the JSON reproduces that wiring as `$source`, `$seed`, `$result`, and `step_id.port`, and it reproduces the cache filename of each intermediate file. The task tmp files themselves are not in this checkout.

## Execution

`run()` executes `plan.steps` in order. Each task receives the files `build()` chose. A task is skipped when its output files already exist, unless `stable_files_override=True`.

```python
reports = pipe.run()
for report in reports:
    print(report.task_name, report.status, report.duration)
```

Replacing `pipe.tasks[i]` requires another `build()` before `run()`. The new plan re-resolves the ports.
