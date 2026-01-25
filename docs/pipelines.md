# Pipeline Generation and Execution

KGpipe allows you to define pipelines manually or using an automatic search algorithm that operates on the PipeKG knowledge base and a set of given constraints. Pipelines are sequences of tasks that transform data from source formats to target knowledge graphs.

## Manual pipeline definition

You can manually define pipelines by creating a `KgPipe` object with a list of tasks. The pipeline will automatically match inputs and outputs between tasks based on their data formats.

```python
from kgpipe.common import KgPipe, Data
from kgpipe.common.models import DataFormat
from kgpipe.common.registry import Registry
import tempfile
import os

# Get tasks from registry
task1 = Registry.get_task("extraction_task")
task2 = Registry.get_task("mapping_task")
task3 = Registry.get_task("fusion_task")

# Create data directory
data_dir = tempfile.mkdtemp()

# Define input and output data
input_data = Data(
    path=os.path.join(data_dir, "source.nt"),
    format=DataFormat.RDF_NTRIPLES
)
output_data = Data(
    path=os.path.join(data_dir, "result.nt"),
    format=DataFormat.RDF_NTRIPLES
)

# Create pipeline
pipe = KgPipe(
    tasks=[task1, task2, task3],
    seed=input_data,
    data_dir=data_dir
)

# Build execution plan
plan = pipe.build(source=input_data, result=output_data)

# Execute pipeline
reports = pipe.run()
```

The `build()` method creates an execution plan that matches task inputs and outputs based on their format specifications. The `run()` method executes each task in sequence, passing outputs from one task as inputs to the next.

## Automatic pipeline generation

Pipelines can be automatically generated from YAML configuration files. The configuration specifies a list of task names and optional configuration parameters:

```yaml
my_pipeline:
    description: "Extract and align RDF data"
    config:
        ENTITY_MATCHING_THRESHOLD: "0.99"
        RELATION_MATCHING_THRESHOLD: "0.5"
    tasks:
        - paris_entity_matching
        - paris_exchange
        - fusion_first_value
        - type_inference_ontology_simple
```

You can load and build pipelines from such configuration files:

```python
from kgpipe.generation.loaders import load_pipeline_catalog, build_from_conf
from kgpipe.common import Data
from pathlib import Path

# Load pipeline configuration
pipeline_confs = load_pipeline_catalog(Path("pipeline.conf"))
pipeline_conf = pipeline_confs.root["my_pipeline"]

# Build pipeline from configuration
target_data = Data(path="target.nt", format=DataFormat.RDF_NTRIPLES)
pipe = build_from_conf(pipeline_conf, target_data, data_dir="/tmp/pipeline_data")

# Build and run
pipe.build(source=input_data, result=output_data)
pipe.run()
```

The automatic pipeline generation uses the PipeKG to find compatible tasks and create valid pipelines based on input/output format constraints.

## Pipeline execution

Once a pipeline is built, you can execute it with the `run()` method. The pipeline will:

1. Execute each task in sequence
2. Pass outputs from one task as inputs to the next
3. Track execution status and generate reports for each task
4. Skip tasks if their outputs already exist (unless `stable_files_override=True`)

```python
# Execute pipeline
reports = pipe.run()

# Check results
for report in reports:
    print(f"Task: {report.task_name}, Status: {report.status}, Duration: {report.duration}")
```

You can also swap individual tasks or subpipelines to experiment with different approaches:

```python
# Replace a task in the pipeline
new_task = Registry.get_task("alternative_task")
pipe.tasks[1] = new_task

# Rebuild and run
pipe.build(source=input_data, result=output_data)
pipe.run()
```

## Pipeline planning

The `build()` method creates an execution plan (`KgPipePlan`) that specifies:
- The sequence of tasks to execute
- Input and output files for each task
- Data format matching between tasks

The plan ensures that each task receives inputs in the correct format and produces outputs that can be consumed by subsequent tasks. If format matching fails, the build process will raise an error indicating which task cannot be connected.
