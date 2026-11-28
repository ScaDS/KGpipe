# Pipeline specification and execution

A pipeline catalog is a user specification. `KgPipe.build()` resolves it into a `KgPipePlan`: concrete files, the port each file was bound to, the parameter values, and a fingerprint of that candidate. The catalog does not itself store the task DAG.

## Catalog

`experiments/examples/pipeline.conf` is a catalog written in this form. Its tasks are the dummies in `experiments/examples/src/kgpipe_examples/dummy_tasks.py`. `experiments/moviekg/pipeline.conf` is the working MovieKG catalog: a description, an optional legacy `config` map, and an ordered list of task names. `connect` defaults to `auto`.

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

Load a catalog and build a pipeline with:

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
