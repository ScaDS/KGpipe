# Adopting KGpipe (integrating existing pipelines/tools)

This page explains how to **adopt KGpipe** when you already have:
- an existing KG pipeline (e.g., DBpedia-style multi-step workflows), and/or
- existing implementations you want to reuse (Python code, Dockerized tools, external APIs).

The goal is to map “what you already have” onto KGpipe’s building blocks:
- **Tasks**: reusable steps with typed inputs/outputs (`input_spec` / `output_spec`)
- **Pipelines**: ordered task graphs (`KgPipe`) that transform `Data` from seed → result
- **Configuration**: parameters passed into tasks (often via env/config profiles)

## 1) Convert an existing pipeline into a KGpipe pipeline

When you have a pipeline described elsewhere (scripts, Airflow, Makefile, DBpedia extraction steps, etc.), do this:

1. **List pipeline steps** (one row per step): name, inputs, outputs, and “how it runs” (Python/Docker/API).
2. **Define formats** for each boundary artifact (RDF formats, CSV, JSON, text). If needed, extend formats.
3. **Wrap each step as a KGpipe task** (see sections below).
4. **Compose tasks into a `KgPipe`** and verify the input/output formats connect.

Practical tip: start by wrapping a *single* step and run it via `kgpipe task ...`, then grow into a pipeline.

## 2) Wrap existing tasks (three common patterns)

### A) Wrap a Dockerized CLI tool

Use this when the tool is a command-line program and can run inside a container.

Reference example:
- `src/kgpipe_tasks/entity_resolution/matcher/paris_rdf_matcher.py`

What to document for each wrapper:
- Docker image name + how to build/pull it
- command template (mapping KGpipe input/output keys to CLI args)
- volume mounts / working dir assumptions
- required environment variables

### B) Wrap existing Python code

Use this when you have Python functions/classes you want to call directly.

Reference example:
- `experiments/param-opti/src/param_opti/tasks/base_linker.py`

What to document for each wrapper:
- the function/class you call
- how you read from `inputs[...]` and write to `outputs[...]`
- how you map configuration parameters into function args (or config objects)

### C) Wrap an external API (HTTP service)

Use this when the implementation is “some service endpoint” (DBpedia Spotlight, LLM providers, etc.).

Reference examples:
- `experiments/param-opti/src/param_opti/tasks/spotlight_lib.py`
- `experiments/param-opti/src/param_opti/tasks/spotlight.py`

What to document for each wrapper:
- endpoint URL + auth
- request/response format
- retry/timeouts and caching
- how you handle rate limits and partial failures

## 3) Discovery (making your tasks available)

Once tasks exist in a Python package, KGpipe can discover them (they register when imported).

```bash
kgpipe discover --package <your_package> --show-results
kgpipe list --type tasks
```

## 4) Recommended structure for “adopted” pipelines

A maintainable layout usually separates:
- `tasks/`: wrappers (Python/Docker/API)
- `pipelines/`: composition (KgPipe builders or pipeline configs)
- `configs/`: pipeline/task configuration profiles
- `docker/`: Dockerfiles and wrapper scripts (if needed)

## Status

This page is the intended replacement for `migration.md` (which was a misleading name). It will be expanded with
copy-pastable code snippets for each wrapper type using the referenced files above as canonical examples.

