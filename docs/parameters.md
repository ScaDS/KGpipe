# Parameters

Task parameters are declared on `KgTask.config_spec` as `Parameter` objects. A pipeline catalog supplies the values. The resolved values are part of the candidate fingerprint. See [Configuration](configuration.md) for the catalog fields and [Pipelines](pipelines.md) for how a plan is built.

## Declaration

```python
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType

ConfigurationDefinition(
    name="Paris entity matching configuration",
    parameters=[
        Parameter(
            name="entity matching threshold",
            datatype=ParameterType.number,
            default_value=0.9,
            native_keys=["ENTITY_MATCHING_THRESHOLD"],
        ),
    ],
)
```

The YAML key is the normalized name: lowercase, with spaces and hyphens turned into single underscores. `entity matching threshold` is `entity_matching_threshold`. `native_keys` adds further aliases, including the legacy environment-style names.

## Binding

For each step whose task declares `config_spec`, values are taken in this order:

1. `default_value` on the parameter
2. the legacy `config` map (`ENTITY_MATCHING_THRESHOLD` and the other aliases on the configuration page)
3. the pipeline `parameters` map, applied to every task that has that parameter
4. `parameters` on that step

The step value wins. The resolved map is stored on the plan step under the normalized names. A task function that takes `config` receives the `ConfigurationProfile`. A function that only takes `inputs` and `outputs` ignores the profile; tasks that still read `os.environ` keep that behavior until they accept `config`.

A pipeline-level parameter that matches nothing, or a step parameter the task does not declare, fails before execution.

## What is not a parameter

Choosing which task implementation runs is the `tasks` list, not a parameter. The port wiring is `connect: auto` or the explicit `inputs` / `outputs` refs. Both are covered by `plan.fingerprint` together with the parameter values. File paths are not.

## Configuration space

A configuration space is a pipeline layout plus the tasks that may fill it. The layout is an ordered list of categories. A task covers the categories declared on `KgTask.category`, unless the space names a different set of slots for that task.

Each searchable parameter has a domain. The domain is `Parameter.allowed_values`, or a shorter list. Every domain value has to match the parameter's type and stay inside `allowed_values`, `minimum`, and `maximum`. `ConfigurationSpace.from_tasks` builds this from the live tasks. `ConfigurationSpace.from_legacy` reads the paper's `{task_key: {category, parameter: [values]}}` dicts and checks those lists against the same constraints.

`enumerate_task_combinations` lists the task orders the layout accepts. A task is used for the current category only when that category is one of its slots and none of its slots are already covered. A task that covers several slots satisfies the later ones. `sample_candidate` draws one accepted order and one value from each domain. The same `random.Random` seed draws the same candidate.

Search strategies stay in `experiments/param-opti`. They call this space through `kgpipe_search.configuration`.

## Legacy search snapshots

Published parameter-search results are `{task_keys, profiles}` JSON files. The file name, without `.json`, is the SHA-256 hex digest of `json.dumps(snapshot, sort_keys=True)`. That digest is the result key. `plan.fingerprint` identifies the resolved execution plan and does not rename those files.

`read_legacy_snapshot` loads one file and checks the digest. `iter_legacy_result_snapshots` reads a directory of those files. It skips `*.plan.json`, `*.eval.json`, and `results.json`. The four published directories are `parameter_search/rdf_sampled`, `parameter_search/text_sampled`, `parameter_search/rdf`, and `parameter_search/text`. The last two are the exhaustive runs. A local copy lives at `data/kgpipe-experiment-results/parameter_search/`.

Versioned fixture files are a different wrapper, `{version: 1, samples: [snapshot, ...]}`. `load_legacy_sample_file` reads that list. The experiment loaders (`load_rdf_sampled_pipeline_configs`, and the text, unique, and exhaustive variants) read the same wrapper.

Loading a snapshot does not change it. `candidate_from_snapshot` keeps the profile names and the binding order, and `candidate_snapshot` writes the same `{task_keys, profiles}` object, so the digest stays the same. `build_candidate` then turns that candidate into a `kgpipe.pipeline/v1` pipeline. Rebuilding the pipeline produces the same fingerprint as a fixed catalog with the same tasks, port refs, and parameter values.
