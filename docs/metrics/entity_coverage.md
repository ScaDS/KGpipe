# Entity Coverage Metric

The Entity Coverage metric evaluates how well source entities are integrated into the target knowledge graph. It measures the overlap between expected source entities and the entities actually present in the generated knowledge graph.

## Source Entity Integration Score

The metric compares a set of expected source entities (provided as a reference file) against the entities found in the knowledge graph. It calculates coverage based on entity URIs and labels.

## Input Format

The expected entities are provided in a CSV or JSON file with the following structure:

**CSV Format:**
```
URI, LABEL, TYPE
http://example.org/entity1, "Entity Label 1", EntityType
http://example.org/entity2, "Entity Label 2", EntityType
```

**JSON Format:**
```json
{
  "http://example.org/entity1": {
    "entity_label": "Entity Label 1",
    "entity_type": "EntityType"
  },
  "http://example.org/entity2": {
    "entity_label": "Entity Label 2",
    "entity_type": "EntityType"
  }
}
```

## Calculation

The metric performs the following steps:

1. **Load expected entities**: Reads the entity dictionary from the provided file path
2. **Extract entity identifiers**: Collects URIs and labels from the expected entities
3. **Find entities in KG**: Searches the knowledge graph for entities matching by URI or label (using `rdfs:label`)
4. **Calculate overlap**: Counts how many expected entities are found in the KG

The coverage score is calculated as:

```
coverage = overlapping_entities_count / expected_entities_count
```

Where:
- `overlapping_entities_count`: Number of expected entities found in the KG
- `expected_entities_count`: Total number of entities in the reference file

## Variants

The framework provides several variants of entity coverage metrics:

- **SourceEntityCoverageMetric**: Strict matching by URI and label
- **SourceEntityCoverageMetricSoft**: Fuzzy matching using label embeddings (threshold 0.95)
- **SourceTypedEntityCoverageMetric**: Matching based on entity type pairs, calculating precision and recall on entity-type combinations

## Usage

To use this metric in evaluation, provide the path to the verified source entities file in the reference configuration:

```python
from kgpipe.evaluation.aspects.reference import ReferenceConfig

config = ReferenceConfig(
    VERIFIED_SOURCE_ENTITIES="path/to/entities.csv"
)
```

The metric will automatically be included when evaluating with the `REFERENCE` aspect.
