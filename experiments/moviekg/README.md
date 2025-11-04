# Inc Movie KG

Documentation and experiment code for incremental KG generation and evaluation.


# Dataset Overview

- 📊 [Benchmark Datasets](https://doi.org/10.5281/zenodo.17246357)

A benchmark derived from Wikipedia and DBpedia in the movie domain covering the three entities: `Film,Person,Company` described and connected by 23(+2) attributes.
The dataset consists of the following.

Four Splits and three different formats:
- RDF: RDF from DBpedia, in the three namespaces for seed, reference and source data
- JSON: json files built from the tree like subgraphs of each film
- TEXT: abstract text of each film entity from wikipedia

Suplmenetary data:
- reference entity matches: for entity matching eval (rdf, json)
- reference entity links: for entity linking eval (text)
- provannce mappings: for tracing json entity mappings
- refernce key mappings: for tracing json to rdf schema matching

Available in three sizes:
- small 100 films: for development
- medium 1,000 films: for testing
- large 10,000 films: for benchmarking

# Running

It is possible to execute the experiemnt in a docker environment.
Adapt the `docker.env` file 
and choose the dataset size (small, medium, large)

> LLM tasks are disabled by default to enable them add
> make pipelines-llm as task in [moviekg_docker.sh](../../scripts/moviekg_docker.sh)

Prepare
```
make setup_docker
```

Execution of dataset stats, pipelines, evalaution, and paper content generation
```
make run_docker_small
```

For more detailed information see also [reproduce.md](../../docs/reproduce.md) or [docs](../../docs/)

# Directory Structure

## Input Structure

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

## Output Structure

```
├── small
│   ├── all_metrics.csv
│   ├── json_a
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
│   ├── json_b[... trunc]
│   ├── paper
│   │   ├── test_fig....png
│   │   └── test_tab.....png
└── medium[... trunc]
```