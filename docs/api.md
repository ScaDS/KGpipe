# Python API

`Pipeline` is the public API for a fixed pipeline. It accepts a typed `kgpipe.pipeline/v1` specification, resolves a `KgPipePlan`, and returns a `PipelineRun` from `run()`. YAML and Python definitions compile through the same `PipelineConf` and `KgPipe.build()`, so equivalent definitions share one fingerprint.

```python
from kgpipe import Pipeline

pipeline = Pipeline(
    name="rdf",
    description="Align source RDF with a seed KG",
    tasks=[
        "paris_entity_matching",
        "paris_exchange",
        "fusion_first_value",
    ],
    parameters={"entity_matching_threshold": 0.99},
    source="data/source.nt",
    seed="data/seed.nt",
    result="output/result.nt",
    data_dir="output",
)

plan = pipeline.plan()
print(plan.fingerprint)
for step in plan.steps:
    print(step.id, step.task, step.inputs, step.outputs, step.parameters)

outcome = pipeline.run()
print(outcome.status, outcome.result.path)
for report in outcome.reports:
    print(report.task_name, report.status, report.duration)
```

`validate()` is the same resolution as `plan()` and does not execute tasks. `run()` plans first. An unknown task, a bad parameter, a backend the task does not declare, or ports that cannot be wired raise `CatalogValidationError` and name the pipeline. A task that fails, and a missing input file, come back as `outcome.status == "failed"` with `outcome.error` set. Task reports stay on `outcome.reports`.

`outcome.status` is `success`, `skipped`, or `failed`. `skipped` means every task was skipped because its outputs were already on disk. `duration` is the wall time of that execution. `plan` on the result is the plan that ran.

## Typed configuration

A task entry is a registry name, a step mapping, a `PipelineTaskConf`, or a live `KgTask`. A live task does not have to be registered. `parameters` are checked against each task's `config_spec`: type, allowed values, and range. Declared defaults are part of the resolved plan. A pipeline parameter that matches no task is an error.

`from_conf` takes the catalog model directly:

```python
from kgpipe import Pipeline, PipelineConf, PipelineTaskConf

conf = PipelineConf(
    description="Align source RDF with a seed KG",
    parameters={"entity_matching_threshold": 0.99},
    tasks=[
        PipelineTaskConf(
            id="match",
            task="paris_entity_matching",
            parameters={"entity_matching_threshold": 0.8},
        ),
        "paris_exchange",
        "fusion_first_value",
    ],
)
pipeline = Pipeline.from_conf(
    "rdf",
    conf,
    source="data/source.nt",
    seed="data/seed.nt",
    result="output/result.nt",
    data_dir="output",
)
plan = pipeline.validate()
```

Port declarations use the same shape as the catalog. A format string is shorthand for `{role: <port>, format: <format>}`.

```python
pipeline = Pipeline(
    tasks=["paris_entity_matching"],
    inputs={"source": "nt", "seed": "nt"},
    output={"result": "paris.csv"},
    source="data/source.nt",
    seed="data/seed.nt",
    result="output/matches.paris.csv",
    data_dir="output",
)
```

`connect` is `auto` unless a step binds ports. Explicit bindings use the same refs as the catalog: `$source`, `$seed`, `$result`, and `step_id.port`. `seed` and `target` are the same input. Relative paths resolve against `base_dir`, or the current directory when `base_dir` is omitted. `backend` is `python`, `docker`, or `remote`.

`pipeline.conf` is the `PipelineConf` that was compiled. File paths stay on the plan, not in the fingerprint.

## YAML

`Pipeline.from_yaml` loads one catalog entry and plans it. A file with a single pipeline does not need a name. Arguments override `defaults` in the pipeline, which override `defaults` in the catalog. `${VAR}` and `${VAR:-default}` expand the same way as the loader. See [Pipelines](pipelines.md).

```python
from kgpipe import Pipeline

pipeline = Pipeline.from_yaml(
    "pipeline.yaml",
    "rdf_base",
    source="data/source.nt",
    seed="data/seed.nt",
    result="output/result.nt",
    data_dir="output",
)
plan = pipeline.plan()
outcome = pipeline.run()
```

A Python definition and that YAML entry produce the same fingerprint when the task order, step ids, port refs, and parameter values agree. `experiments/examples/src/kgpipe_examples/api_example.py` is the Python form of `experiments/examples/sample.yaml`.

## Migration from KgPipe

`KgPipe` still builds and runs. `Pipeline` is the specification in front of it. `to_kgpipe()` returns the compiled execution object after `plan()` or `run()`.

```python
from kgpipe.common import Data, KgPipe

pipe = KgPipe(tasks=tasks, seed=seed, data_dir=data_dir)
pipe.build(source=source, result=result)
reports = pipe.run()
```

The same run through the public API:

```python
from kgpipe import Pipeline

pipeline = Pipeline(
    tasks=tasks,
    source=source,
    seed=seed,
    result=result,
    data_dir=data_dir,
)
plan = pipeline.plan()
outcome = pipeline.run()
reports = outcome.reports
```

`Pipeline.from_kgpipe(pipe)` wraps a `KgPipe` that already has a plan and keeps that fingerprint. Pass `source` and `result` only when `build()` has not been called yet; the adapter then calls `build()` on that same object. Planning a new `Pipeline` from the wrapper's `.conf` follows the catalog rules, including declared defaults.

A new `Pipeline` binds declared parameter defaults into the plan, which is what a YAML catalog does. A `KgPipe` built by hand, with no `config_catalog`, leaves `parameters` empty. Use `from_kgpipe` when that existing plan has to stay as it is. Use `Pipeline(...)` or `from_conf` when the run should match the catalog.

`build_from_yaml` and `validate_pipeline` remain the loader. `Pipeline.from_yaml` and `Pipeline.validate` call them.
