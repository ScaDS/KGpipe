# KGpipe: A Framework for Knowledge Graph Integration Pipelines

- 📊 [Benchmark Datasets](https://doi.org/10.5281/zenodo.17246357)


KGpipe is an open-source framework for defining, executing, and evaluating knowledge graph (KG) integration pipelines.
It enables the reuse and composition of existing tools (e.g., OpenIE, PARIS, JedAI) and Large Language Models (LLMs) into modular pipelines that integrate heterogeneous data sources into a unified KG.

**Key features:**
- Modular and extensible pipeline specification.
- Support for multiple execution backends (Python, Docker, HTTP services).
- Standardized I/O between tasks for reproducibility and interoperability.
- Novel benchmark for systematic evaluation of pipelines across RDF, JSON, and text sources.
- Metrics covering structural, semantic, and reference-based evaluation.

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
TODO
