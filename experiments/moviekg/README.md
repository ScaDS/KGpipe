# MovieKG (KGpipe pipelines)

This directory contains **MovieKG pipeline definitions and execution helpers** for running incremental KG construction
pipelines with KGpipe.

Evaluation of the produced KGs is now handled in the **KGI-Bench** repository (Movie benchmark). See:
- `KGI-Bench/docs/reproduce.md`
- `KGI-Bench/docs/cli.md` (includes `kgibench evaluate --benchmark movie ...`)

## What’s in here

- **Pipeline catalog**: `pipeline.conf` (pipeline variants and their task sequences)
- **Execution helpers**: `src/moviekg/pipelines/` (pytest-driven runners + helpers)
- **Environment templates**: `env`, `docker_env` (copy to `.env` / `docker.env` for local configuration)

## Running pipelines (local)

From `experiments/moviekg/`:

```bash
cp env .env
make pipelines
```

LLM variants:

```bash
make pipelines-llm
```

Per-pipeline targets are also available (see `Makefile`), e.g.:

```bash
make test-json-base
make test-rdf-base
make test-msp-all
```

## Running pipelines (Docker workflow)

This uses the `Makefile` targets to build images + start services and run pipelines inside Docker.

```bash
cp docker_env docker.env
make setup_docker
make run_docker_small
```

> Note: LLM pipelines are typically disabled by default in Docker orchestration; enable them by adding the
> `pipelines-llm` step to the orchestration script used in your setup.

## Dataset overview (high level)

- Dataset release: `https://doi.org/10.5281/zenodo.17246357`
- Sizes: `small` (100 films), `medium` (1k), `large` (10k)
- Formats per split: RDF, JSON, TEXT (incremental splits with seed/reference/source)

## Directory structure

### Input structure (example)

```
├── film_100
│   ├── entities
│   │   └── master_entities.csv
│   ├── ontology.ttl -> ../movie-ontology.ttl
│   ├── split_0
│   │   ├── index
│   │   │   └── entities.csv
│   │   ├── kg
│   │   │   ├── reference
│   │   │   │   ├── data/
│   │   │   │   ├── data_agg.nt
│   │   │   │   ├── data.nt
│   │   │   │   └── meta/
│   │   │   └── seed
│   │   │       ├── data/
│   │   │       ├── data.nt
│   │   │       └── meta/
│   │   └── sources
│   │       ├── json
│   │       │   ├── data/
│   │       │   └── meta/
│   │       ├── rdf
│   │       │   ├── data/
│   │       │   ├── data.nt
│   │       │   └── meta/
│   │       └── text
│   │           ├── data/
│   │           └── meta/
│   ├── split_1[... trunc]
├── film_1k[... trunc]
```

### Output structure (example)

Pipeline outputs are written under `$OUTPUT_DIR/$DATASET_SELECT/<pipeline_name>/stage_<n>/` and include:
- `result.nt` (and optionally `result_eval.nt`)
- `exec-plan.json`, `exec-report.json`
- `tmp/` intermediate artifacts

```
├── small
│   ├── json_base
│   │   ├── stage_1
│   │   │   ├── exec-plan.json
│   │   │   ├── exec-report.json
│   │   │   ├── result.nt
│   │   │   └── tmp/
│   │   ├── stage_2
│   │   │   ├── exec-plan.json
│   │   │   ├── exec-report.json
│   │   │   ├── result.nt
│   │   │   └── tmp/
│   │   └── stage_3
│   │       ├── exec-plan.json
│   │       ├── exec-report.json
│   │       ├── result.nt
│   │       └── tmp/
│   ├── json_alt[... trunc]
└── medium[... trunc]
```