# KG Evaluation

The framework provides several approaches to evaluate the quality of a generated knowledge graph. Evaluation is organized into different aspects, each focusing on specific quality dimensions.

## Evaluation Aspects

The framework supports evaluation across multiple aspects:

- **Statistical**: Basic metrics like triple count, entity count, graph density, and other structural properties
- **Semantic**: Validation of ontology consistency, type errors, relation direction, and semantic correctness
- **Reference**: Comparison against curated gold-standard knowledge graphs using precision, recall, and F1 scores
- **Efficiency**: Resource consumption metrics including runtime, memory usage, and cost

## Using the Evaluator

The main entry point for evaluation is the `Evaluator` class. You configure which aspects to evaluate and then run evaluation on a knowledge graph:

```python
from kgpipe.evaluation import Evaluator, EvaluationConfig, EvaluationAspect
from kgpipe.common.models import KG, DataFormat
from pathlib import Path

# Create evaluation configuration
config = EvaluationConfig(
    aspects=[EvaluationAspect.STATISTICAL, EvaluationAspect.SEMANTIC, EvaluationAspect.REFERENCE],
    metrics=None  # None means use all available metrics for each aspect
)

# Create evaluator
evaluator = Evaluator(config)

# Load the knowledge graph to evaluate
kg = KG(
    id="my_kg",
    name="My Knowledge Graph",
    path=Path("result.nt"),
    format=DataFormat.RDF_NTRIPLES
)

# For reference-based evaluation, provide reference data
references = {
    "gold_standard": Data(path=Path("gold_standard.nt"), format=DataFormat.RDF_NTRIPLES)
}

# Run evaluation
report = evaluator.evaluate(kg, references=references)

# Access results
print(f"Overall score: {report.overall_score}")
for aspect_result in report.aspect_results:
    print(f"{aspect_result.aspect.value}: {len(aspect_result.metrics)} metrics")
    for metric in aspect_result.metrics:
        print(f"  {metric.name}: {metric.value} (normalized: {metric.normalized_score})")
```

## Evaluation via CLI

You can also evaluate knowledge graphs using the command-line interface:

```bash
kgpipe eval target.nt --ground-truth gold.nt --aspects statistical semantic reference --output results.json
```

The CLI supports:
- `--aspects`: Specify which aspects to evaluate (statistical, semantic, reference, efficiency)
- `--metrics`: Filter to specific metrics by name
- `--ground-truth`: Path to reference knowledge graph for reference-based evaluation
- `--output`: Save evaluation results to a JSON file

## Statistical Evaluation

Statistical evaluation provides basic metrics about the knowledge graph structure:

- Triple count
- Entity count
- Relation count
- Graph density
- Average degree
- Connected components

These metrics help understand the scale and structure of the generated knowledge graph.

## Semantic Evaluation

Semantic evaluation validates the knowledge graph against its ontology:

- Disjoint domain violations
- Incorrect relation direction
- Incorrect relation cardinality
- Incorrect relation domain/range
- Incorrect datatypes
- Ontology class coverage
- Ontology relation coverage
- Namespace coverage

These metrics ensure the knowledge graph conforms to its schema and maintains semantic consistency.

## Reference-based Evaluation

Reference-based evaluation compares the generated knowledge graph against a gold standard:

- Entity matching (precision, recall, F1)
- Relation matching (precision, recall, F1)
- Triple alignment
- Source typed entity coverage
- Reference class coverage

This type of evaluation requires a curated reference knowledge graph that serves as ground truth.

## Evaluation Reports

Evaluation results are returned as `EvaluationReport` objects that contain:

- The evaluated knowledge graph
- Reference data used (if any)
- Aspect results for each evaluated aspect
- Individual metric results with values and normalized scores
- Overall score (average of normalized scores across all metrics)

Reports can be serialized to JSON for storage and later analysis:

```python
report.to_json("evaluation_results.json")
```
