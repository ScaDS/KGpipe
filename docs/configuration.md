# Framework configuration

KGpipe has three configuration layers: the framework, a pipeline specification, and the parameter values bound onto tasks.

## Framework

The framework configuration is a YAML file, usually `config.yaml`. It selects the Meta KG connection and the namespaces used for constructed resources:

```yaml
SYS_KG_URL: "memory://"
SYS_KG_USR: ""
SYS_KG_PSW: ""

SOURCE_NAMESPACE: "http://kg.org/rdf/"
TARGET_RESOURCE_NAMESPACE: "http://kg.org/resource/"
TARGET_ONTOLOGY_NAMESPACE: "http://kg.org/ontology/"
```

`SYS_KG_URL` is `memory://`, a SQLite path, or a SPARQL endpoint. It is loaded when the framework starts:

```python
from kgpipe.common.config import load_config

config = load_config()
print(config.SYS_KG_URL)
```

## Pipeline specification

A catalog entry names the tasks and, optionally, their parameter values. See [Pipelines](pipelines.md) for how `build()` connects ports.

```yaml
rdf_base:
    description: "Align source RDF with target KG"
    parameters:
        entity_matching_threshold: 0.99
        relation_matching_threshold: 0.5
    tasks:
        - paris_entity_matching
        - fusion_first_value
```

`parameters` applies to every task whose `config_spec` has that parameter. A step can override it:

```yaml
tasks:
    - name: paris_entity_matching
      parameters:
          entity_matching_threshold: 0.8
```

Building a pipeline binds these values into a `ConfigurationProfile` and stores them on the resolved plan. `run()` passes that profile to the task when the task function accepts `config`. Values are checked against the declared type, allowed values, and range before execution.

A pipeline parameter that matches no task `config_spec` is an error. A step parameter on a task that does not declare `config_spec` is an error.

## Legacy `config` map

MovieKG catalogs still carry a string map:

```yaml
config:
    ENTITY_MATCHING_THRESHOLD: "0.99"
    RELATION_MATCHING_THRESHOLD: "0.5"
    LLM_MODEL: "gpt-5-mini"
```

`build_from_conf` binds these keys onto each task that declares them. `fusion_first_value` reads `entity_matching_threshold` and `relation_matching_threshold` from that profile. The MovieKG runner still copies `LLM_MODEL` into `DEFAULT_LLM_MODEL_NAME` for the LLM tasks.

When a task does declare `config_spec`, the same keys are also bound as parameters:

| Legacy key | Parameter |
|---|---|
| `ENTITY_MATCHING_THRESHOLD` | `entity_matching_threshold` |
| `RELATION_MATCHING_THRESHOLD` | `relation_matching_threshold` |
| `LLM_MODEL` | `llm_model` |

Names match after lowercasing and turning spaces or hyphens into underscores, so a parameter declared as `entity matching threshold` receives `entity_matching_threshold`. `native_keys` on `Parameter` is the other alias. Unrecognized legacy keys are left for the caller; they do not fail the catalog load.

Binding order is: the parameter default, legacy `config`, pipeline `parameters`, then the step's `parameters`. The newer fields override the legacy map.

## Ontology

`fusion_first_value` and `type_inference_ontology_simple` still read `ONTOLOGY_PATH`. The CLI stores `--ontology` on the pipeline object. It is not yet a `$ontology` port in `build()`.
