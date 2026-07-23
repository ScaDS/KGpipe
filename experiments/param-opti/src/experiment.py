#!/usr/bin/env python3
"""
Run a full configuration search experiment.

Given a search space (RDF or text), a search strategy proposes pipeline configs;
each candidate is executed against seed/source data, evaluated against a reference
KG, and written to the output directory. A combined results file is produced for
offline analysis via analyse.py.

Example:
    PYTHONPATH=src python src/experiment.py \\
        --seed data/bench/.../seed/data.nt \\
        --source data/bench/.../sources/rdf/data.nt \\
        --reference data/bench/.../reference/data_agg.nt \\
        --ontology data/bench/.../ontology.ttl \\
        --pipeline-type rdf \\
        --strategy qgns \\
        --budget 20 \\
        --init-budget 3 \\
        --output-dir fix_runs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from kgpipe_search.configuration import (
    pipeline_config_to_snapshot,
    print_pipeline_config_short,
    task_keys_from_pipeline_config,
)
from kgpipe_search.definitions import (
    RDF_PIPELINE_LAYOUT,
    RDF_SEARCH_SPACE,
    TEXT_PIPELINE_LAYOUT,
    TEXT_SEARCH_SPACE,
    PipelineConfig,
)
from kgpipe_search.evaluation import evaluate_pipeline
from kgpipe_search.search import (
    bayesian_optimization,
    hnr_search,
    hnr_2_search,
    implementation_aware_search,
    llm_search,
    qgns_search,
    random_search,
)
from kgpipe_search.strategies.strategies import SearchRun

import execute as pipeline_execute

PipelineType = Literal["rdf", "text"]
SearchStrategyName = Literal["random", "implementation_aware", "qgns", "hnr", "hnr_2", "bayesian", "llm"]
TasksTmpScope = Literal["config", "pipeline", "shared"]
InitStrategy = Literal["random", "implementation_aware"]


def _pipeline_context(pipeline_type: PipelineType) -> tuple[Dict[str, Any], Any, Callable[..., Path]]:
    if pipeline_type == "rdf":
        return RDF_SEARCH_SPACE, RDF_PIPELINE_LAYOUT, pipeline_execute.run_rdf_pipeline
    if pipeline_type == "text":
        return TEXT_SEARCH_SPACE, TEXT_PIPELINE_LAYOUT, pipeline_execute.run_text_pipeline
    raise ValueError(f"Unsupported pipeline type {pipeline_type!r}")


def _run_search(
    *,
    strategy: SearchStrategyName,
    budget: int,
    evaluate_fn: Callable[[PipelineConfig], float],
    search_space: Dict[str, Any],
    pipeline_layout: Any,
    init_budget: int,
    init_strategy: InitStrategy,
    y: int,
    k: int,
    rho: float,
    pool_size: int,
    beta: float,
    llm_max_retries: int,
    rng: random.Random,
) -> SearchRun:
    common = {
        "budget": budget,
        "evaluate_fn": evaluate_fn,
        "search_space": search_space,
        "pipeline_layout": pipeline_layout,
        "rng": rng,
    }

    if strategy == "random":
        return random_search(**common)

    if strategy == "implementation_aware":
        return implementation_aware_search(**common, y=y)

    if strategy == "qgns":
        return qgns_search(
            **common,
            init_budget=init_budget,
            init_strategy=init_strategy,
            y=y,
            k=k,
            rho=rho,
        )

    if strategy == "hnr":
        if init_budget <= 0:
            raise ValueError("HNR requires --init-budget > 0")
        return hnr_search(
            **common,
            init_budget=init_budget,
            init_strategy=init_strategy,
            y=y,
            rho=rho,
        )

    if strategy == "hnr_2":
        return hnr_2_search(
            **common,
            init_budget=init_budget,
            init_strategy=init_strategy,
            y=y,
            rho=rho,
        )
    if strategy == "bayesian":
        return bayesian_optimization(
            **common,
            init_random=init_budget,
            init_strategy=init_strategy,
            y=y,
            pool_size=pool_size,
            beta=beta,
        )

    if strategy == "llm":
        return llm_search(
            **common,
            max_retries=llm_max_retries,
        )

    raise ValueError(f"Unknown search strategy {strategy!r}")


def run_search_experiment(
    *,
    seed_path: Path,
    source_path: Path,
    reference_path: Path,
    ontology_path: Optional[Path],
    output_dir: Path,
    pipeline_type: PipelineType,
    strategy: SearchStrategyName,
    budget: int,
    init_budget: int,
    init_strategy: InitStrategy,
    y: int,
    k: int,
    rho: float,
    pool_size: int,
    beta: float,
    llm_max_retries: int,
    rng_seed: int,
    tasks_tmp_scope: TasksTmpScope,
    results_path: Optional[Path],
    reuse_existing: bool = True,
) -> Dict[str, Any]:
    pipeline_execute._set_ontology_env(ontology_path)

    search_space, pipeline_layout, run_pipeline = _pipeline_context(pipeline_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(rng_seed)
    run_results: List[Dict[str, Any]] = []
    search_history: List[Dict[str, Any]] = []
    best_score: Optional[float] = None
    cache_hits = 0

    def evaluate_fn(pipeline_config: PipelineConfig) -> float:
        nonlocal best_score, cache_hits

        task_keys = task_keys_from_pipeline_config(pipeline_config)
        snapshot = pipeline_config_to_snapshot(task_keys, pipeline_config)
        config_hash = pipeline_execute._config_hash(snapshot)

        config_path = output_dir / f"{config_hash}.json"
        result_path = output_dir / f"{config_hash}.nt"
        eval_path = output_dir / f"{config_hash}.eval.json"
        plan_path = output_dir / f"{config_hash}.plan.json"
        tasks_tmp_dir = pipeline_execute._tasks_tmp_dir(
            output_dir=output_dir,
            config_hash=config_hash,
            task_keys=task_keys,
            scope=tasks_tmp_scope,
        )

        step = len(run_results) + 1
        print(f"\n=== trial {step}/{budget} ({config_hash}) ===")
        print_pipeline_config_short(pipeline_config)

        pipeline_execute._write_config_snapshot(config_path, snapshot)

        entry: Dict[str, Any] = {
            "trial": step,
            "config_hash": config_hash,
            "config_path": str(config_path),
            "result_path": str(result_path),
            "eval_path": str(eval_path),
            "plan_path": str(plan_path),
            "tasks_tmp_dir": str(tasks_tmp_dir),
            "status": "ok",
            "cached": False,
        }

        try:
            cached = pipeline_execute._load_cached_eval(eval_path) if reuse_existing else None
            if cached is not None:
                cache_hits += 1
                entry["cached"] = True
                entry["status"] = cached["status"]
                if cached["status"] == "error":
                    entry["error"] = cached["error"]
                    score = float(cached["score"])
                    print(f"cached error: {entry['error']}")
                else:
                    entry["evaluation"] = cached["evaluation"]
                    score = float(cached["score"])
                    print(f"cached score: {score:.6f}")
            elif reuse_existing and result_path.exists():
                aggregate_score = evaluate_pipeline(
                    pipeline_config,
                    result_path,
                    reference_path,
                )
                evaluation = pipeline_execute._to_jsonable(aggregate_score)
                entry["evaluation"] = evaluation
                entry["cached"] = "result_only"
                pipeline_execute._write_eval_snapshot(eval_path, evaluation)
                score = float(aggregate_score.final_score)
                print(f"reused result, score: {score:.6f}")
            else:
                run_pipeline(
                    pipeline_config,
                    seed_path=seed_path,
                    source_path=source_path,
                    result_path=result_path,
                    plan_path=plan_path,
                    tasks_tmp_dir=tasks_tmp_dir,
                    run_name=config_hash,
                )
                aggregate_score = evaluate_pipeline(
                    pipeline_config,
                    result_path,
                    reference_path,
                )
                evaluation = pipeline_execute._to_jsonable(aggregate_score)
                entry["evaluation"] = evaluation
                pipeline_execute._write_eval_snapshot(eval_path, evaluation)
                score = float(aggregate_score.final_score)
                print(f"score: {score:.6f}")
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            score = 0.0
            pipeline_execute._write_eval_snapshot(
                eval_path,
                {"status": "error", "error": entry["error"]},
            )
            print(f"failed: {entry['error']}")

        run_results.append(entry)
        best_score = score if best_score is None else max(best_score, score)
        return score

    print(f"Running search strategy={strategy!r} budget={budget}")
    print(f"seed: {seed_path}")
    print(f"source: {source_path}")
    print(f"reference: {reference_path}")
    print(f"output_dir: {output_dir}")
    print(f"tasks_tmp_scope: {tasks_tmp_scope}")

    search_run = _run_search(
        strategy=strategy,
        budget=budget,
        evaluate_fn=evaluate_fn,
        search_space=search_space,
        pipeline_layout=pipeline_layout,
        init_budget=init_budget,
        init_strategy=init_strategy,
        y=y,
        k=k,
        rho=rho,
        pool_size=pool_size,
        beta=beta,
        llm_max_retries=llm_max_retries,
        rng=rng,
    )

    result_by_hash = {item["config_hash"]: item for item in run_results}
    running_best: Optional[float] = None

    for step, ((score, _cfg), decision) in enumerate(
        zip(search_run.history, search_run.decisions),
        start=1,
    ):
        task_keys = task_keys_from_pipeline_config(_cfg)
        snapshot = pipeline_config_to_snapshot(task_keys, _cfg)
        config_hash = pipeline_execute._config_hash(snapshot)
        running_best = score if running_best is None else max(running_best, score)

        entry = result_by_hash.get(config_hash, {})
        search_history.append(
            {
                "step": step,
                "decision": decision,
                "config_hash": config_hash,
                "score": score,
                "best_score": running_best,
                "status": entry.get("status", "unknown"),
                "config_path": entry.get("config_path"),
                "result_path": entry.get("result_path"),
                "eval_path": entry.get("eval_path"),
                "plan_path": entry.get("plan_path"),
            }
        )

    payload: Dict[str, Any] = {
        "pipeline_type": pipeline_type,
        "search": {
            "strategy": strategy,
            "budget": budget,
            "init_budget": init_budget,
            "init_strategy": init_strategy,
            "y": y,
            "k": k,
            "rho": rho,
            "pool_size": pool_size,
            "beta": beta,
            "rng_seed": rng_seed,
            "decisions": search_run.decisions,
        },
        "seed": str(seed_path),
        "source": str(source_path),
        "reference": str(reference_path),
        "ontology": str(ontology_path) if ontology_path is not None else None,
        "output_dir": str(output_dir),
        "tasks_tmp_scope": tasks_tmp_scope,
        "results": run_results,
        "search_history": search_history,
        "best_score": running_best,
        "cache_hits": cache_hits,
        "reuse_existing": reuse_existing,
    }

    resolved_results_path = results_path or (output_dir / "results.json")
    resolved_results_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote combined results to {resolved_results_path}")

    succeeded = sum(1 for item in run_results if item["status"] == "ok")
    print(
        f"Finished: {succeeded}/{len(run_results)} succeeded, "
        f"cache_hits={cache_hits}, best_score={running_best}"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a full pipeline configuration search experiment.",
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
        default=Path("data/tmp/search_runs"),
        help="Directory for pipeline outputs, eval files, and task temp files",
    )
    parser.add_argument(
        "--pipeline-type",
        choices=["rdf", "text"],
        default="rdf",
        help="Pipeline family to search over",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "implementation_aware", "qgns", "hnr", "hnr_2", "bayesian", "llm"],
        default="random",
        help=(
            "Search strategy to use. "
            "'random' = uniform random configs; "
            "'implementation_aware' = systematic task-combo coverage with random params"
        ),
    )
    parser.add_argument("--budget", type=int, default=10, help="Total number of configs to evaluate")
    parser.add_argument(
        "--init-budget",
        type=int,
        default=3,
        help="Initialization budget for qgns/hnr/bayesian (ignored by random)",
    )
    parser.add_argument(
        "--init-strategy",
        choices=["random", "implementation_aware"],
        default="implementation_aware",
        help="Initialization sampling strategy",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=1,
        help="Number of parameter samples per task combo during implementation-aware init",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k anchors for QGNS")
    parser.add_argument(
        "--rho",
        type=float,
        default=0.2,
        help="Exploration probability for QGNS/HNR",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=32,
        help="Candidate pool size for Bayesian optimization",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Acquisition beta for Bayesian optimization",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Validation retries per LLM proposal when using --strategy llm",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="RNG seed for reproducible search",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to write combined results JSON (default: <output-dir>/results.json)",
    )
    parser.add_argument(
        "--tasks-tmp-scope",
        choices=["config", "pipeline", "shared"],
        default="config",
        help=(
            "How to name/reuse the per-run tasks tmp dir: "
            "'config' = one tmp dir per config hash (default), "
            "'pipeline' = reuse tmp dir for configs with identical task list, "
            "'shared' = reuse one tmp dir for all configs"
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run pipelines and evaluation even when cached result/eval files exist",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.budget <= 0:
        raise SystemExit("--budget must be > 0")
    if args.init_budget < 0:
        raise SystemExit("--init-budget must be >= 0")
    if args.strategy == "hnr" and args.init_budget <= 0:
        raise SystemExit("HNR requires --init-budget > 0")

    seed_path = pipeline_execute._validate_input_path(args.seed, "Seed graph")
    source_path = pipeline_execute._validate_input_path(args.source, "Source input")
    reference_path = pipeline_execute._validate_input_path(args.reference, "Reference graph")
    ontology_path = (
        pipeline_execute._validate_input_path(args.ontology, "Ontology")
        if args.ontology is not None
        else None
    )

    payload = run_search_experiment(
        seed_path=seed_path,
        source_path=source_path,
        reference_path=reference_path,
        ontology_path=ontology_path,
        output_dir=args.output_dir,
        pipeline_type=args.pipeline_type,
        strategy=args.strategy,
        budget=args.budget,
        init_budget=args.init_budget,
        init_strategy=args.init_strategy,
        y=args.y,
        k=args.k,
        rho=args.rho,
        pool_size=args.pool_size,
        beta=args.beta,
        llm_max_retries=args.llm_max_retries,
        rng_seed=args.rng_seed,
        tasks_tmp_scope=args.tasks_tmp_scope,
        results_path=args.results,
        reuse_existing=not args.force_rerun,
    )

    failed = sum(1 for item in payload["results"] if item["status"] != "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
