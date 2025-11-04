# Rep Experiments

Guidelines to run the [experiments](../experiments)
- see also [moviekg](../experiments/moviekg/README.md)

## Overview

1. Prerequesits
2. Running Docker Only
3. Running

## Prerequesits

- 32-64GB memory, 50GB disk space, (2GB nvidia GPU)
- git, make, tar, g(un)zip, docker, (uv)

## Running Docker Only (recommended)

In this mode the KGpipe framework itself is running in docker and needed tasks that are using docker are called by it using the host docker sock as a mount.

Minimal example docker.env file for configuring the experiment in docker, like locations.

Copy `docker_env` to `docker.env` when finished

```ini
# .env
PIPELINE_CONFIG=pipeline.conf

DATASET_SELECT=small

# path inside docker
ONTOLOGY_PATH=/app/experiments/moviekg/movie-ontology.ttl

EMBEDDER=sentence-transformer
EMBED_CACHE="redis://cache:6379"

DBPEDIA_ANNOTATE_URL='http://dbpedia-spotlight:80/rest/annotate'
OPENAI_TOKEN="INSERT_YOUR_OPENAI_TOKEN_HERE"
DEFAULT_LLM_MODEL_NAME=gpt-5-mini

# OPTIONAL
OLLAMA_TOKEN="empty"
LLM_ENDPOINT_URL=https://example.com/ollama/api/generate
```

Prepare
```
make setup_docker
```

Execution of dataset stats, pipelines, evalaution, and paper content generation
```
make run_docker_small
```

## Running (WIP)

1. Pipelines
2. Evaluation
3. Paper Content

### Pipelines

To execute all pipelines run

```
make pipelines
```

Outputs are in `$OUTPUT/${pipeline_name}`

### Evaluation

To create statistics for

```
make evaluation
```

Outputs are in `$OUTPUT/${pipeline_name}_metrics.csv`
and aggregated metrics in `$OUTPUT/all_metrics.csv`

### Paper Content

Creates the figures and tables for the papers.
Not in latex format therefore I recommend using https://www.latex-tables.com/

```
make paper
```

Outputs are in `$OUTPUT/paper`