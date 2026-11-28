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
