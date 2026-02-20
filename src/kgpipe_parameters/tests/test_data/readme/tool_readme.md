# EntityMatcher

A tool for matching entities across knowledge graphs.

## Installation

```bash
pip install entity-matcher
```

## Usage

```bash
entity-matcher --input data.nt --output results.tsv --threshold 0.8 --max-iter 10
entity-matcher --format csv --verbose
```

## Configuration

The following parameters can be set:

- `threshold`: The matching threshold, a float between 0 and 1 (default: 0.5)
- `max_iter`: Maximum number of iterations (default: 10)
- `input`: Path to input knowledge base (required)
- `output`: Path to output results file (required)
- `format`: Output format, one of csv, tsv, json (default: tsv)
- `similarity_metric`: Similarity metric to use, e.g. jaccard, cosine (default: jaccard)

## Advanced Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch_size` | int | Number of entities per batch |
| `num_threads` | int | Number of parallel threads |
| `cache_dir` | path | Directory for caching intermediate results |
| `log_level` | string | Logging level: DEBUG, INFO, WARNING, ERROR |

## Environment Variables

You can also configure via environment:

```bash
export MATCHER_THRESHOLD=0.8
export MATCHER_MAX_MEMORY=4096
```

## Running with Java Backend

For the Java backend, you may need to increase JVM memory:

```bash
java -Xmx8192m -Xms2048m -jar entity-matcher.jar <kb1> <kb2> <outputFolder>
```

## API

See the [API documentation](docs/api.md) for details.

