# Framework Configuration

KGpipe supports configuration at multiple levels: main framework configuration, pipeline configuration, and task-specific configuration parameters.

## Main Configuration

The main framework configuration is stored in a YAML file (typically `config.yaml` in the configuration directory). The configuration includes settings for the Meta KG connection and namespace definitions:

```yaml
SYS_KG_URL: "memory://"
SYS_KG_USR: ""
SYS_KG_PSW: ""

SOURCE_NAMESPACE: "http://kg.org/rdf/"
TARGET_RESOURCE_NAMESPACE: "http://kg.org/resource/"
TARGET_ONTOLOGY_NAMESPACE: "http://kg.org/ontology/"
```

The `SYS_KG_URL` specifies the connection string for the Meta KG (PipeKG). It can be:
- `memory://` for in-memory storage
- A file path for SQLite-based storage
- A SPARQL endpoint URL for remote RDF stores

The configuration is loaded automatically when the framework initializes. You can access it programmatically:

```python
from kgpipe.common.config import load_config

config = load_config()
print(config.SYS_KG_URL)
```

## Pipeline Configuration

When defining pipelines in YAML files, you can specify configuration parameters that will be passed to tasks during execution:

```yaml
my_pipeline:
    description: "Entity resolution pipeline with custom thresholds"
    config:
        ENTITY_MATCHING_THRESHOLD: "0.99"
        RELATION_MATCHING_THRESHOLD: "0.5"
        LLM_MODEL: "gpt-4"
    tasks:
        - paris_entity_matching
        - fusion_first_value
```

These configuration values are made available to tasks as environment variables during execution. Tasks can access them using:

```python
import os

threshold = float(os.getenv("ENTITY_MATCHING_THRESHOLD", "0.5"))
```

## Task Configuration

When defining a task, you can specify configuration parameters that the task expects. These parameters can be:

- **Confidence thresholds**: Used by matching and fusion tasks to determine when to accept or reject results
- **Training parameters**: Used by machine learning-based tasks (e.g., epochs, training size)
- **Algorithm selection**: Some tasks support multiple algorithms that can be selected via configuration
- **Model parameters**: For LLM-based tasks, you can specify which model to use

The framework passes configuration from the pipeline definition to individual tasks through environment variables. This allows tasks to be configured without modifying their code.

## Configuration Extraction

The framework automatically extracts configuration from pipeline definitions and makes it available during task execution. When you build a pipeline from a YAML configuration, the `config` section is processed and environment variables are set before each task runs.

You can also override configuration values programmatically when building pipelines:

```python
from kgpipe.generation.loaders import build_from_conf
import os

# Override configuration
os.environ["ENTITY_MATCHING_THRESHOLD"] = "0.95"

# Build and run pipeline
pipe = build_from_conf(pipeline_conf, target_data, data_dir)
pipe.build(source=input_data, result=output_data)
pipe.run()
```
