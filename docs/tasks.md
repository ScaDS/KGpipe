# Task Specification

A KG pipeline is the combination of multiple tasks: extraction, mapping, matching, fusion. Tasks are the building blocks of pipelines, each performing a specific transformation on data with well-defined input and output formats.

You can define tasks using the `Registry.task` decorator. Tasks can be implemented directly in Python, executed in Docker containers, or call remote services. When you define a task using the decorator, it is automatically registered in the framework's registry and added to the Meta KG.

## Example: Task in pure Python

When you have existing Python libraries or want to implement custom logic directly, you can create a pure Python task. The task function receives dictionaries of input and output `Data` objects, where each `Data` object has a `path` attribute pointing to the file and a `format` attribute indicating the data format.

```python
from kgpipe.common.registry import Registry
from kgpipe.common.models import DataFormat, Data
from typing import Dict

@Registry.task(
    input_spec={"input": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Simple RDF transformation task"
)
def my_python_task(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    Process input RDF and write to output.
    
    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """
    input_path = inputs["input"].path
    output_path = outputs["output"].path
    
    # Your processing logic here
    # Read from input_path, process, write to output_path
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        # Process data...
        f_out.write(f_in.read())
```

**When to use**: Pure Python tasks are ideal when you have existing Python libraries (like rdflib, pandas, transformers) or need custom logic that doesn't require external tools.

## Example: Task using Docker environment

For existing software with command-line interfaces, you can wrap them in Docker containers. KGpipe will use your local Docker environment to execute these tasks.

**Wrapping the tool**

The easiest way is to wrap the tool with a simple shell script that handles the command-line arguments:

```bash
#!/bin/bash
# wrapper.sh
mytool $1 $2 $3
```

Then define a task that uses `run_docker_container_task`:

```python
from kgpipe.common.registry import Registry
from kgpipe.common.models import DataFormat, Data
from kgpipe.common.io import run_docker_container_task
from typing import Dict

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.PARIS_CSV},
    description="Paris entity matching using Docker container"
)
def paris_entity_matching(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    Paris entity matching task that runs in a Docker container.
    
    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """
    result = run_docker_container_task(
        inputs=inputs,
        outputs=outputs,
        image="kgt/paris:latest",
        command_template=["bash", "paris.sh", "{source}", "{kg}", "{output}"]
    )
    return result
```

The `command_template` uses placeholders like `{source}`, `{kg}`, and `{output}` that correspond to keys in your `inputs` and `outputs` dictionaries. These will be automatically replaced with the container paths when the task executes.

**When to use**: Docker tasks are ideal for existing software tools that have command-line interfaces and can be containerized.

## Example: Task using remote service

For services accessible via HTTP APIs, you can call them directly within your task definition:

```python
from kgpipe.common.registry import Registry
from kgpipe.common.models import DataFormat, Data
from typing import Dict
import requests

@Registry.task(
    input_spec={"text": DataFormat.TEXT},
    output_spec={"triples": DataFormat.RDF_NTRIPLES},
    description="Extract triples using remote OpenIE service"
)
def remote_extraction_task(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    Extract triples from text using a remote API service.
    
    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """
    text_path = inputs["text"].path
    output_path = outputs["triples"].path
    
    # Read input text
    with open(text_path, 'r') as f:
        text_content = f.read()
    
    # Call remote API
    response = requests.post(
        "https://api.example.com/extract",
        json={"text": text_content}
    )
    response.raise_for_status()
    
    # Write results to output
    with open(output_path, 'w') as f:
        f.write(response.text)
```

**When to use**: Remote service tasks are useful for calling LLM APIs, web services, or any HTTP-accessible processing endpoints.

## Task discovery

Tasks are automatically registered when their module is imported. To make tasks available to the framework, you need to discover them using one of the following methods:

**CLI**

Discover tasks from a module path:

```bash
kgpipe discover -m path/to/your/tasks.py
```

Or discover from a package:

```bash
kgpipe discover -p your_package_name
```

You can also discover from multiple sources and show results:

```bash
kgpipe discover --all --show-results
```

**Python**

In Python, simply import the module containing your tasks:

```python
import your_tasks_module  # This will trigger task registration
```

When you import a module with `@Registry.task` decorated functions, they are automatically registered in the framework's registry and added to the Meta KG.

## Extending data formats

You can extend the framework with custom data formats by registering them with the `FormatRegistry`. See the example in `experiments/examples/src/kgpipe_examples/config.py` for how to define and register custom formats.
