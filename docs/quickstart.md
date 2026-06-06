# KGpipe Quickstart

This quickstart shows the **current** workflow for defining and running:
- tasks (Python functions registered in the `Registry`)
- pipelines (a `KgPipe` connecting tasks via input/output `Data`)
- metrics/evaluations (evaluators + metrics run on a `KG`)

## See also
- `experiments/examples/`: a minimal example project using KGpipe
- `docs/reproduce.md`: running the (deprecated but working) reproduction experiments

## Install

From the repo root:

```bash
pip install -e .
```

If you need the optional ML stack (transformers / sentence-transformers), install extras:

```bash
pip install -e ".[ml]"
```

## Create a new experiment (recommended starting point)

The easiest way to get a working project layout is to copy the template in `experiments/examples/`.

```bash
cd experiments/examples
./init.sh
```

The script creates a new directory containing a Python package with example tasks/pipelines. Then:

```bash
cd "<your-new-experiment-dir>"
pip install -e .
```

## Define tasks

Tasks are normal Python callables registered via `@Registry.task(...)`. See
`experiments/examples/src/kgpipe_examples/task_examples.py` for canonical examples.

Key concepts:
- **`input_spec` / `output_spec`**: the expected formats for inputs/outputs
- **`TaskInput` / `TaskOutput`**: dict-like objects mapping names to `Data`
- **`trace_task_run`**: wraps the function to produce a run report

Minimal pattern (simplified from the examples):

```python
from kgpipe.common import TaskInput, TaskOutput, trace_task_run
from kgpipe.common.registry import Registry

@trace_task_run
@Registry.task(
    input_spec={"input": "some_format"},
    output_spec={"output": "some_other_format"},
    description="Example task",
)
def my_task(inputs: TaskInput, outputs: TaskOutput):
    outputs["output"].path.touch()
```

## Define and run a pipeline (Python API)

Pipelines connect tasks by passing `Data` (path + format) between them. A minimal example exists in
`experiments/examples/src/kgpipe_examples/pipe_examples.py`.

The core pattern:

```python
from kgpipe.common import KgPipe, Data

# tasks = [task_a, task_b, ...]  # registered task callables (from your package)
# seed = Data(path=..., format=...)
# result = Data(path=..., format=...)
pipe = KgPipe(tasks=tasks, seed=seed, data_dir="/tmp/my_run_dir")
pipe.build(source=seed, result=result)
pipe.run()
```

## Discover components and inspect what’s available (CLI)

The CLI entrypoint is `kgpipe` (see `pyproject.toml`).

To register tasks/pipelines/metrics from your local package, import it via discovery:

```bash
# From inside your experiment venv / environment
kgpipe discover --package <your_python_package> --show-results
```

You can also discover from a local module path (directory or file):

```bash
kgpipe discover --module-path ./src/<your_python_package> --show-results
```

To list what KGpipe currently knows about (after discovery):

```bash
kgpipe list --type tasks
kgpipe list --type metrics
```

To show details for a specific task:

```bash
kgpipe show --type task <task_name>
```

To print YAML templates for evaluation configs:

```bash
kgpipe show metric-config-templates
```

## Run a single task (CLI)

KGpipe can execute a registered task directly. The `--input/--output` syntax is:

\[
\texttt{<path>|<format>@<short\_name>}
\]

(`@<short_name>` is optional.)

Example:

```bash
kgpipe task <task_name> \
  --input  "/tmp/in.txt|txt@input" \
  --output "/tmp/out.txt|txt@output"
```

Tip: if you get “Task not found”, run `kgpipe discover ...` first.

## Run a minimal evaluation / metrics (Python API, new `kgpipe_eval`)

KG evaluation is being migrated to the **new** `kgpipe_eval` package (recommended). A realistic integration-style
example exists in:

- `experiments/moviekg/src/moviekg/evaluation/test_eval_refactor.py`

Minimal example (basic statistics):

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
```