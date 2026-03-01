# KGpipe framework documentation

KGpipe is a framework to define pipelines for data integration into knowledge graphs. The framework enables you to compose existing tools and implementations into modular pipelines that integrate heterogeneous data sources into a unified knowledge graph.

The framework is organized into three main subpackages: `kgpipe` contains the core framework functionality including CLI, common utilities, execution backends, and evaluation components. `kgpipe_tasks` provides task implementations for cleaning, construction, entity resolution, schema alignment, and text processing. `kgpipe_llm` includes LLM-based task implementations and utilities.

## Meta KG

[link](metakg.md)

KGpipe uses an internally maintained Meta KG (PipeKG) to maintain tasks, tool implementations, their components, pipelines, and metrics. This knowledge base enables automatic pipeline generation and tracking of execution results.

## Task Specification

[link](tasks.md)

The framework enables the description and integration of integration tasks. You can describe tasks with Python, interface existing implementations with Python, Docker, or remote API requests. Tasks are discovered and registered through the framework's discovery mechanism.

## Pipeline Generation and Execution

[link](pipelines.md)

KGpipe allows you to define pipelines manually or using an automatic search algorithm that operates on the PipeKG knowledge base and a set of given constraints. You can swap subpipelines or single tasks with other components to experiment with different approaches.

## Configuration

[link](configuration.md)

The framework supports configuration at multiple levels. The main configuration is specified in `kgpipe.yml`, and individual tasks can define their own configuration parameters that will be passed by the framework when executing pipelines.

## Evaluation

[link](evaluation.md)

The framework provides several approaches to evaluate the quality of a generated knowledge graph, including accuracy, coverage, consistency, statistics, and efficiency measurements. Evaluation metrics are tracked in the Meta KG alongside pipeline results.

Additional evaluation metrics are documented in the [metrics](metrics/) directory, such as [entity coverage](metrics/entity_coverage.md).

## Other Links

- [Reproducing the movie kg experiments for 15 pipelines](reproduce.md) (rdf, json, text)

## Docu Backlog

- Explain different execution modes
    - File Batches
    - Streaming
- Explain advanced pipelines
- Ontology creation...