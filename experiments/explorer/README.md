# KGpipe Explorer

A static web application for exploring the KGpipe framework's System Knowledge Graph (PipeKG) and pipeline execution results. The explorer provides an interactive interface to browse registered tasks, pipelines, metrics, and evaluation results without executing pipelines.

## Overview

The KGpipe Explorer is designed to visualize and navigate the meta knowledge graph that KGpipe maintains internally. This System KG tracks:

- **Tasks**: Registered integration tasks with their specifications, input/output formats, and categories
- **Pipelines**: Pipeline definitions and their composition of tasks
- **Metrics**: Evaluation metrics and quality measurements
- **Execution Results**: Results from pipeline runs and their associated metadata

## Purpose

The explorer enables users to:

- Discover available tasks and their capabilities
- Understand pipeline structures and task dependencies
- Review evaluation metrics and execution results
- Explore relationships between tasks, pipelines, and data formats
- Navigate the System KG structure through an intuitive interface

## System Knowledge Graph

The explorer operates on the PipeKG (Meta Knowledge Graph) that KGpipe maintains internally. For detailed information about the System KG structure, query capabilities, and SPARQL examples, see the [Meta KG documentation](../../docs/metakg.md).

## Design Principles

- **Static**: The explorer works with pre-generated System KG data and execution results. It does not execute pipelines or modify the framework state.
- **Read-only**: All exploration is read-only, ensuring no accidental modifications to pipeline definitions or execution results.
- **Interactive**: Provides an intuitive interface for navigating the complex relationships in the System KG.

## Architecture

The explorer consumes static RDF data from the System KG and presents it through a web-based interface, allowing users to query and visualize the knowledge graph structure without requiring direct SPARQL knowledge.

## Backlog
- decide on framwork and src structure