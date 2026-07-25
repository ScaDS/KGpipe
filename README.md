# KGpipe: A Framework for Knowledge Graph Integration Pipelines

KGpipe is an open-source framework for defining, executing, and evaluating knowledge graph (KG) integration pipelines.
It enables the reuse and composition of existing tools (e.g., OpenIE, PARIS, JedAI) and Large Language Models (LLMs) into modular pipelines that integrate heterogeneous data sources into a unified KG.

![KGpipe workflow](docs/workflow.png)

## Related benchmarks, datasets, and papers

- [**KGI-Bench**](https://github.com/ScaDS/KGI-Bench): benchmark specification + tooling for KG integration evaluation.
- [**KGI-Bench (Movies)**](https://doi.org/10.5281/zenodo.17246357): Movie-domain benchmark dataset release (Zenodo).
- [**KGpipe Explorer**](https://vehnem.github.io/kgpipe-explorer/): a demo exploring results of KGI-Bench executed with KGpipe.
- [**Framework Paper**](https://arxiv.org/abs/2511.18364): framework core paper; revised version accepted at QDB 2026 (to appear).

**Who is this for?**
- You have multiple heterogeneous sources (RDF/JSON/text) and want a **reproducible, modular pipeline**.
- You want to **reuse existing tooling** (Python libs, Dockerized CLIs, remote APIs/LLMs) without rewriting everything.
- You want to **evaluate** generated KGs with a growing set of metrics (`kgpipe_eval`).

**Key features:**
- Modular and extensible pipeline specification.
- Support for multiple execution backends (Python, Docker, HTTP services).
- Standardized I/O between tasks for reproducibility and interoperability.
- Novel benchmark for systematic evaluation of pipelines across RDF, JSON, and text sources.
- Metrics covering structural, semantic, and reference-based evaluation.

## Quickstart (5 minutes)

Install from source (editable):

```bash
pip install -e .
kgpipe --help
```

Bootstrap a minimal example project and discover its tasks:

```bash
cd experiments/examples
./init.sh

cd "<your-new-experiment-dir>"
pip install -e .

kgpipe discover --package <your_python_package> --show-results
kgpipe list --type tasks
```

## Architecture

Each pipeline is a sequence of tasks with well-defined input/output contracts.
Execution backends supported:

- Python functions (e.g., using rdflib, transformers).
- Docker containers (for legacy or external tools).
- HTTP services (remote APIs, LLM endpoints).

Pipelines are executed sequentially with file-based I/O to ensure logging, debugging, and cross-language compatibility


## Core Integration Tasks

- Knowledge Extraction (KE): Extract triples from raw text or JSON.
- Data Mapping (DM): Map extracted data to target ontology.
- Ontology/Schema Matching (OM/SA): Align classes and relations.
- Entity Resolution (ER): Detect equivalent entities.
- Entity Fusion (EF): Merge aligned entities and attributes.
- Data Cleaning (DC) & Completion (KC): Ensure consistency and enrich missing data

## Pipelines

KGpipe provides Single-Source Pipelines (SSPs) and Multi-Source Pipelines (MSPs):
**SSPs**: Incrementally integrate sources of the same type (RDF, JSON, or text).
**MSPs**: Combine sources across different formats.

## Evaluation Metrics

- Statistical Metrics – triple count, entity count, graph density.
- Resource Metrics – runtime, memory, cost.
- Semantic Validation – ontology consistency, type errors, relation direction.
- Reference Validation – fidelity against curated gold-standard KGs

## Usage

Documentation lives in `docs/`:
- **Start here**: `docs/index.md` and `docs/quickstart.md`
- **Adopting KGpipe / wrapping existing tools**: `docs/adoption.md`
- **Evaluation (new API)**: `docs/evaluation.md` (uses `kgpipe_eval`)
- **MovieKG reproduction**: `docs/reproduce.md`

### Documentation site (GitHub Pages)

This repo is set up to build docs with **MkDocs + Material**:
- config: `mkdocs.yml`
- local build instructions: `docs/README.md`
- deploy workflow: `.github/workflows/docs.yml` (GitHub Pages via Actions)

## Installation notes (CPU vs CUDA)

Some optional ML dependencies (e.g. `sentence_transformers`) pull in PyTorch (`torch`). Depending on which PyTorch wheel gets selected, you may see large downloads like `nvidia-*` and `triton`.

KGpipe keeps the ML stack out of the default install; install it explicitly when needed. For `uv`, PyTorch is pinned to the official PyTorch wheel indexes to avoid accidentally pulling CUDA wheels from PyPI.

### Base install (fast, no torch)

```bash
uv pip install .
```

### ML install with CPU-only PyTorch (no `nvidia-*`)

```bash
uv pip install ".[ml,cpu]"
```

### ML install with CUDA-enabled PyTorch (will download `nvidia-*`)

```bash
uv pip install ".[ml,cuda]"
```

## Experiments
- **[moviekg](experiments/moviekg/README.md)** evaluation of pipelines, building a Movie KG from three sources (rdf, json, text).
