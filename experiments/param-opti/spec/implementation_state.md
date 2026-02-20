# Implementation State

Last updated: 2026-02-20

## Parameter Extractors

| Source Type     | Regex | LLM | Tests | Module                          |
|-----------------|-------|-----|-------|---------------------------------|
| CLI help        | ✅    | ✅  | ✅    | `extractors/cli.py`             |
| Python lib      | ✅    | ✅  | ✅    | `extractors/python_lib.py`      |
| HTTP API doc    | ✅    | ✅  | ✅    | `extractors/http_api.py`        |
| Docker doc      | ✅    | ✅  | ✅    | `extractors/docker.py`          |
| Repo README/Doc | ✅    | ✅  | ✅    | `extractors/readme_doc.py`      |

All extractors live under `src/kgpipe_parameters/extraction/extractors/`.

## Core Infrastructure

| Component                  | Status | Location                                      |
|----------------------------|--------|-----------------------------------------------|
| Models                     | ✅     | `extraction/models.py`                        |
| Base classes               | ✅     | `extraction/base.py`                          |
| Regex patterns             | ✅     | `extraction/patterns.py`                      |
| Utilities                  | ✅     | `extraction/utils.py`                         |
| ParameterMiner             | ✅     | `extraction/param_miner.py`                   |
| Auto source detect         | ✅     | `param_miner._detect_source_type()`           |
| Keyword chunk filter       | ✅     | `extraction/chunk_filter.py`                  |

## Keyword Chunk Filter

Keyword-based pre-filter that scores chunks before they reach any extractor.
Counts parameter-signal keywords per language/file-type and skips files below
a configurable threshold.  No embeddings, zero extra dependencies.

| Language / Type | Keywords cover                                          | Threshold |
|-----------------|---------------------------------------------------------|-----------|
| Python          | argparse, click, dataclass, Field, os.environ, …       | 2         |
| Java            | @Option, @Parameter, getProperty, Properties, @Value,… | 1         |
| .properties     | `=`, `:`                                                | 1         |
| XML             | `<property`, `<param`, `<config`, `name=`, `value=`, … | 1         |
| Docker          | ENV, ARG, EXPOSE, environment:, …                      | 2         |
| README          | --, default, parameter, configuration, usage, …        | 1*        |
| Generic         | default, param, config, option, argument, …            | 2         |

\* README files use threshold 1 in `extract_from_repo()`.

**Tests:** `tests/test_chunk_filter.py` — 29 tests.

## Experiment Pipeline

| Step                             | Status | Location                              |
|----------------------------------|--------|---------------------------------------|
| Tool discovery from input/       | ✅     | `experiment.py` + `tool.py`           |
| Repository cloning               | ✅     | `experiment.py`                       |
| CLI extraction                   | ✅     | `experiment.py:extract_from_cli()`    |
| README extraction (input)        | ✅     | `experiment.py:extract_from_readme()` |
| Repo file extraction             | ✅     | `experiment.py:extract_from_repo()`   |
| Keyword chunk filter in pipeline | ✅     | `experiment.py:extract_from_repo()`   |
| JSON output + summary            | ✅     | `experiment.py:save_result()`         |

### File types scanned by `extract_from_repo()`

| File type      | Extractor used    | Chunk filter | Limit |
|----------------|-------------------|--------------|-------|
| `*.py`         | PYTHON_LIB        | ✅ (≥2)      | 20    |
| `*.java`       | README (kv+flags) | ✅ (≥1)      | 20    |
| `*.properties` | README (kv)       | inherent     | 15    |
| `*.xml` (cfg)  | README            | ✅ (≥1)      | 10    |
| `Dockerfile*`  | DOCKER            | —            | all   |
| `docker-compose*` | DOCKER         | —            | all   |
| `README*`, `*.md`, docs/ | README  | ✅ (≥1)      | 15    |

## Parameter Clustering

Cluster similar parameters across tools using sentence-transformer embeddings
and agglomerative clustering.

| Component                | Status | Location                                       |
|--------------------------|--------|-------------------------------------------------|
| Clustering models        | ✅     | `clustering/models.py`                          |
| Embedding similarity     | ✅     | `clustering/similarity.py`                      |
| Agglomerative clusterer  | ✅     | `clustering/clusterer.py`                        |
| Cluster __init__         | ✅     | `clustering/__init__.py`                         |
| Experiment integration   | ✅     | `experiment.py:cluster_parameters()`             |
| CLI flags                | ✅     | `--cluster`, `--cluster-only`, `--distance-threshold` |
| JSON + CSV output        | ✅     | `_clusters.json`, `_parameter_table.csv`         |
| Tests                    | ✅     | `tests/test_clustering.py` — 20 tests            |

**Approach:**
1. Load extracted parameters from per-tool JSON output files.
2. Build text representation (name + description + native_keys + type_hint).
3. Encode with `all-MiniLM-L6-v2` sentence-transformer.
4. L2-normalise embeddings; compute cosine distance matrix.
5. Apply scikit-learn `AgglomerativeClustering` (average linkage, precomputed
   distance, configurable threshold — default 0.55).
6. Label each cluster by most-common parameter name.
7. Output `_clusters.json` (full cluster details) and `_parameter_table.csv`
   (flat, one row per parameter with cluster assignment).

## Still TODO

- [ ] Visualization (`visualization/kgpipe_parameter_explorer.py` is a stub)
- [ ] Optimization (`optimization/` is empty)
- [ ] Embedding-based RAG for LLM prompts (later, when keyword filter plateaus)
