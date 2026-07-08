#!/usr/bin/env python3
"""
Run and evaluate pipeline configs from fixture files.

Example (quick test with the small sampled fixture, 6 RDF / 4 text configs):
    python experiment.py \\
        --seed data/bench/.../seed/data.nt \\
        --source data/bench/.../sources/rdf/data.nt \\
        --reference data/bench/.../reference/data_agg.nt \\
        --ontology data/bench/.../ontology.ttl

Full exhaustive run (all task/parameter permutations):
    python experiment.py ... --configs exhaustive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kgpipe.common import Data, DataFormat, KgPipe
from kgpipe_search.configuration import (
    load_rdf_exhaustive_pipeline_configs,
    load_rdf_sampled_pipeline_configs,
    load_text_exhaustive_pipeline_configs,
    load_text_sampled_pipeline_configs,
    pipeline_config_to_snapshot,
    print_pipeline_config_short,
    task_keys_from_pipeline_config,
)
from kgpipe_search.definitions import PipelineConfig
from kgpipe_search.evaluation import evaluate_pipeline


def _install_param_opti_shim() -> None:
    if "param_opti" in sys.modules:
        return

    param_opti = types.ModuleType("param_opti")
    tasks = types.ModuleType("param_opti.tasks")

    for lib in (
        "base_linker_lib",
        "base_matcher_lib",
        "paris_lib",
        "fusion_lib",
        "spotlight_lib",
        "corenlp_lip",
        "genie_lib",
    ):
        module = import_module(f"kgpipe_search.dev.tasks.{lib}")
        setattr(tasks, lib, module)
        sys.modules[f"param_opti.tasks.{lib}"] = module

    param_opti.tasks = tasks
    sys.modules["param_opti"] = param_opti
    sys.modules["param_opti.tasks"] = tasks


_install_param_opti_shim()


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _set_ontology_env(ontology_path: Optional[Path]) -> None:
    if ontology_path is None:
        return
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
    os.environ["ONTOLOGY_PATH"] = str(ontology_path.resolve())


def _config_hash(snapshot: Dict[str, Any]) -> str:
    canonical = json.dumps(snapshot, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_config_snapshot(config_path: Path, snapshot: Dict[str, Any]) -> None:
    config_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_input_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _load_pipeline_configs(
    *,
    pipeline_type: str,
    configs: str,
    configs_fixture: Optional[Path],
) -> List[PipelineConfig]:
    loaders: Dict[str, Dict[str, Callable[[], List[PipelineConfig]]]] = {
        "rdf": {
            "sampled": load_rdf_sampled_pipeline_configs,
            "exhaustive": load_rdf_exhaustive_pipeline_configs,
        },
        "text": {
            "sampled": load_text_sampled_pipeline_configs,
            "exhaustive": load_text_exhaustive_pipeline_configs,
        },
    }

    if pipeline_type not in loaders:
        raise ValueError(f"Unsupported pipeline type {pipeline_type!r}")
    if configs not in loaders[pipeline_type]:
        raise ValueError(f"Unsupported configs mode {configs!r}")

    loader = loaders[pipeline_type][configs]
    loaded = loader(configs_fixture) if configs_fixture is not None else loader()
    if not loaded:
        raise ValueError(
            f"No pipeline configs loaded for pipeline_type={pipeline_type!r}, configs={configs!r}. "
            "Generate fixtures with the configuration tests first."
        )
    return loaded


def run_rdf_pipeline(
    pipeline_config: PipelineConfig,
    *,
    seed_path: Path,
    source_path: Path,
    result_path: Path,
    tasks_tmp_dir: Path,
    run_name: str,
) -> Path:
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = KgPipe(
        tasks=pipeline_config.tasks,
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=tasks_tmp_dir,
        name=run_name,
    )

    pipeline.build(
        stable_files=True,
        configCatalog=pipeline_config.config_catalog,
        source=Data(path=source_path, format=DataFormat.RDF_NTRIPLES),
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES),
    )
    pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=False)
    return result_path


def run_text_pipeline(
    pipeline_config: PipelineConfig,
    *,
    seed_path: Path,
    source_path: Path,
    result_path: Path,
    tasks_tmp_dir: Path,
    run_name: str,
) -> Path:
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = KgPipe(
        tasks=pipeline_config.tasks,
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=tasks_tmp_dir,
        name=run_name,
    )

    pipeline.build(
        stable_files=True,
        configCatalog=pipeline_config.config_catalog,
        source=Data(path=source_path, format=DataFormat.TEXT),
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES),
    )
    pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=False)
    return result_path


def run_all_configs(
    *,
    seed_path: Path,
    source_path: Path,
    reference_path: Path,
    ontology_path: Optional[Path],
    output_dir: Path,
    pipeline_type: str,
    configs: str,
    configs_fixture: Optional[Path],
    start: int,
    limit: Optional[int],
    results_path: Optional[Path],
) -> List[Dict[str, Any]]:
    _set_ontology_env(ontology_path)

    run_pipeline = run_rdf_pipeline if pipeline_type == "rdf" else run_text_pipeline

    pipeline_configs = _load_pipeline_configs(
        pipeline_type=pipeline_type,
        configs=configs,
        configs_fixture=configs_fixture,
    )

    end = len(pipeline_configs) if limit is None else min(len(pipeline_configs), start + limit)
    selected = pipeline_configs[start:end]

    output_dir.mkdir(parents=True, exist_ok=True)
    run_results: List[Dict[str, Any]] = []

    print(f"Running {len(selected)} pipeline config(s) [{start}:{end})")
    print(f"seed: {seed_path}")
    print(f"source: {source_path}")
    print(f"reference: {reference_path}")
    print(f"output_dir: {output_dir}")

    for offset, pipeline_config in enumerate(selected, start=start):
        task_keys = task_keys_from_pipeline_config(pipeline_config)
        snapshot = pipeline_config_to_snapshot(task_keys, pipeline_config)
        config_hash = _config_hash(snapshot)

        result_path = output_dir / f"{config_hash}.nt"
        config_path = output_dir / f"{config_hash}.json"
        tasks_tmp_dir = output_dir / f"{config_hash}_tasks_tmp"
        run_name = config_hash

        print(f"\n=== config {offset + 1}/{len(pipeline_configs)} ({config_hash}) ===")
        print_pipeline_config_short(pipeline_config)

        _write_config_snapshot(config_path, snapshot)

        entry: Dict[str, Any] = {
            "config_idx": offset,
            "config_hash": config_hash,
            "config_path": str(config_path),
            "result_path": str(result_path),
            "status": "ok",
        }

        try:
            run_pipeline(
                pipeline_config,
                seed_path=seed_path,
                source_path=source_path,
                result_path=result_path,
                tasks_tmp_dir=tasks_tmp_dir,
                run_name=run_name,
            )
            aggregate_score = evaluate_pipeline(
                pipeline_config,
                result_path,
                reference_path,
            )
            entry["evaluation"] = _to_jsonable(aggregate_score)
            print(f"score: {aggregate_score.final_score:.6f}")
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"failed: {entry['error']}")

        run_results.append(entry)

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline_type": pipeline_type,
            "configs": configs,
            "seed": str(seed_path),
            "source": str(source_path),
            "reference": str(reference_path),
            "ontology": str(ontology_path) if ontology_path is not None else None,
            "output_dir": str(output_dir),
            "start": start,
            "limit": limit,
            "results": run_results,
        }
        results_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote scores to {results_path}")

    succeeded = sum(1 for item in run_results if item["status"] == "ok")
    print(f"\nFinished: {succeeded}/{len(run_results)} succeeded")
    return run_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute and evaluate pipeline configs from fixture files.",
    )
    parser.add_argument("--seed", type=Path, required=True, help="Path to seed knowledge graph")
    parser.add_argument("--source", type=Path, required=True, help="Path to source input graph/text")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to reference knowledge graph used for evaluation",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=None,
        help="Optional ontology path (sets ONTOLOGY_PATH for matchers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tmp/pipeline_runs"),
        help="Directory for pipeline outputs and task temp files",
    )
    parser.add_argument(
        "--pipeline-type",
        choices=["rdf", "text"],
        default="rdf",
        help="Pipeline family to run",
    )
    parser.add_argument(
        "--configs",
        choices=["sampled", "exhaustive"],
        default="sampled",
        help=(
            "Which fixture set to execute: "
            "'sampled' = small fixture for quick tests (default), "
            "'exhaustive' = all task/parameter permutations"
        ),
    )
    parser.add_argument(
        "--configs-fixture",
        type=Path,
        default=None,
        help="Optional path to a custom configs fixture JSON file",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index into the loaded config list",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of configs to run (default: all from --start)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to write a single JSON summary of all run scores (default: <output-dir>/results.json)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0")

    seed_path = _validate_input_path(args.seed, "Seed graph")
    source_path = _validate_input_path(args.source, "Source input")
    reference_path = _validate_input_path(args.reference, "Reference graph")
    ontology_path = (
        _validate_input_path(args.ontology, "Ontology")
        if args.ontology is not None
        else None
    )

    run_results = run_all_configs(
        seed_path=seed_path,
        source_path=source_path,
        reference_path=reference_path,
        ontology_path=ontology_path,
        output_dir=args.output_dir,
        pipeline_type=args.pipeline_type,
        configs=args.configs,
        configs_fixture=args.configs_fixture,
        start=args.start,
        limit=args.limit,
        results_path=args.results or (args.output_dir / "results.json"),
    )

    failed = sum(1 for item in run_results if item["status"] != "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
