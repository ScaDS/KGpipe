#!/usr/bin/env python3
"""
Offline analysis of search strategies on already computed pipeline eval results.

This script treats a `results.json` (as written by `experiment.py`) as a cache:
- some configs have an evaluation score (status == "ok")
- some configs are missing or errored (partial results)

We can then "simulate" different search strategies without re-running any pipeline by
letting the strategy propose configs and looking them up in the cache.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

# Optional integration with the existing kgpipe_search strategies.
try:
    from kgpipe_search.configuration import pipeline_config_snapshot_key
    from kgpipe_search.definitions import RDF_PIPELINE_LAYOUT, RDF_SEARCH_SPACE, PipelineConfig
    from kgpipe_search.search import bayesian_optimization, hnr_search, qgns_search, random_search

    _HAS_KGPIPE_SEARCH = True
    _KGPIPE_SEARCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    _HAS_KGPIPE_SEARCH = False
    _KGPIPE_SEARCH_IMPORT_ERROR = exc


class Candidate(NamedTuple):
    config_hash: str
    task_key: Tuple[str, ...]
    snapshot_path: Optional[Path]


class StepLog(NamedTuple):
    step: int
    proposed_hash: str
    hit: bool
    score: Optional[float]
    best_score: Optional[float]
    misses_so_far: int


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_key_from_snapshot_dict(snapshot: Dict[str, Any]) -> str:
    # Must match kgpipe_search.configuration.pipeline_config_snapshot_key()'s serialization:
    # json.dumps(snapshot, sort_keys=True)
    return json.dumps(snapshot, sort_keys=True)


def _load_score_by_snapshot_key(
    *,
    results_path: Path,
    entry_by_hash: Dict[str, Dict[str, Any]],
    score_by_hash: Dict[str, float],
) -> Dict[str, float]:
    """
    Map config snapshots (serialized canonical JSON) -> score.
    This lets us evaluate PipelineConfig objects sampled by kgpipe_search strategies.
    """
    score_by_key: Dict[str, float] = {}
    for h, entry in entry_by_hash.items():
        score = score_by_hash.get(h)
        if score is None:
            continue
        snapshot_path = _resolve_snapshot_path(results_path, entry.get("config_path"))
        if snapshot_path is None:
            continue
        try:
            snap = _read_json(snapshot_path)
            if isinstance(snap, dict):
                key = _snapshot_key_from_snapshot_dict(snap)
                score_by_key[key] = float(score)
        except Exception:
            continue
    return score_by_key


class OfflineCacheOracle:
    def __init__(self, score_by_snapshot_key: Dict[str, float], *, miss_score: float = 0.5) -> None:
        self._score_by_key = score_by_snapshot_key
        self.miss_score = float(miss_score)
        self.hits = 0
        self.misses = 0

    def evaluate(self, cfg: "PipelineConfig") -> float:
        key = pipeline_config_snapshot_key(cfg, RDF_SEARCH_SPACE)
        score = self._score_by_key.get(key)
        if score is None:
            self.misses += 1
            return self.miss_score
        self.hits += 1
        return float(score)

def _load_cache(results_path: Path) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      score_by_hash: config_hash -> final_score (only status == ok)
      entry_by_hash: config_hash -> raw results entry (all statuses)
    """
    payload = _read_json(results_path)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected 'results' list in {results_path}")

    entry_by_hash: Dict[str, Dict[str, Any]] = {}
    score_by_hash: Dict[str, float] = {}

    for item in results:
        if not isinstance(item, dict):
            continue
        h = item.get("config_hash")
        if not isinstance(h, str):
            continue
        entry_by_hash[h] = item
        if item.get("status") == "ok":
            evaluation = item.get("evaluation") or {}
            if isinstance(evaluation, dict) and isinstance(evaluation.get("final_score"), (int, float)):
                score_by_hash[h] = float(evaluation["final_score"])

    return score_by_hash, entry_by_hash


def _resolve_snapshot_path(results_path: Path, raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.is_absolute():
        return p if p.exists() else None
    candidate = results_path.parent / p
    return candidate if candidate.exists() else None


def _load_candidates(
    *,
    results_path: Path,
    entry_by_hash: Dict[str, Dict[str, Any]],
    include_missing_snapshots: bool,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    for h, entry in entry_by_hash.items():
        snapshot_path = _resolve_snapshot_path(results_path, entry.get("config_path"))
        if snapshot_path is None and not include_missing_snapshots:
            continue

        task_key: Tuple[str, ...] = ()
        if snapshot_path is not None:
            try:
                snapshot = _read_json(snapshot_path)
                task_keys = snapshot.get("task_keys")
                if isinstance(task_keys, list) and all(isinstance(x, str) for x in task_keys):
                    task_key = tuple(task_keys)
            except Exception:
                task_key = ()

        candidates.append(Candidate(config_hash=h, task_key=task_key, snapshot_path=snapshot_path))

    return candidates


class Strategy:
    name: str

    def propose(self) -> str:  # returns config_hash
        raise NotImplementedError

    def observe(self, config_hash: str, score: Optional[float]) -> None:
        # score is None for cache miss or non-ok result
        return


class RandomStrategy(Strategy):
    name = "random"

    def __init__(self, rng: random.Random, universe: Sequence[Candidate]) -> None:
        self._rng = rng
        self._universe = universe

    def propose(self) -> str:
        return self._rng.choice(self._universe).config_hash


class GreedyKnownStrategy(Strategy):
    """
    Upper bound / sanity check: picks the best already-known score.
    Useful to verify the harness and to see what "best possible" would be in the cache.
    """

    name = "greedy-known"

    def __init__(self, rng: random.Random, universe: Sequence[Candidate], score_by_hash: Dict[str, float]) -> None:
        self._rng = rng
        self._universe = universe
        self._score_by_hash = score_by_hash
        self._ordered: List[str] = [
            c.config_hash for c in sorted(universe, key=lambda c: score_by_hash.get(c.config_hash, float("-inf")), reverse=True)
        ]
        self._i = 0

    def propose(self) -> str:
        if self._i >= len(self._ordered):
            return self._rng.choice(self._universe).config_hash
        h = self._ordered[self._i]
        self._i += 1
        return h


class UCBByTaskKeyStrategy(Strategy):
    """
    Lightweight bandit baseline:
    - treat each distinct task pipeline (task_keys tuple) as an arm
    - within an arm, sample configs uniformly
    - update arm rewards based on observed scores

    This is robust to partial caches: misses just don't update the arm.
    """

    name = "ucb-taskkey"

    def __init__(self, rng: random.Random, universe: Sequence[Candidate], exploration: float = 2.0) -> None:
        self._rng = rng
        self._exploration = exploration

        arms: Dict[Tuple[str, ...], List[str]] = {}
        for c in universe:
            arms.setdefault(c.task_key, []).append(c.config_hash)
        self._arms = arms
        self._arm_keys = list(arms.keys())

        self._n_total = 0
        self._n: Dict[Tuple[str, ...], int] = {k: 0 for k in self._arm_keys}
        self._mean: Dict[Tuple[str, ...], float] = {k: 0.0 for k in self._arm_keys}

    def propose(self) -> str:
        # Ensure each arm is tried at least once
        for k in self._arm_keys:
            if self._n[k] == 0:
                return self._rng.choice(self._arms[k])

        # Standard UCB1 over arms
        self._n_total = max(1, self._n_total)
        best_k = None
        best_ucb = float("-inf")
        for k in self._arm_keys:
            bonus = math.sqrt((self._exploration * math.log(self._n_total)) / self._n[k])
            ucb = self._mean[k] + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_k = k
        assert best_k is not None
        return self._rng.choice(self._arms[best_k])

    def observe(self, config_hash: str, score: Optional[float]) -> None:
        self._n_total += 1
        if score is None:
            return
        # find arm by scanning (cheap at this scale); if needed we can add hash->arm map later
        for k, hashes in self._arms.items():
            if config_hash in hashes:
                n = self._n[k] + 1
                prev = self._mean[k]
                self._mean[k] = prev + (score - prev) / n
                self._n[k] = n
                return


def _build_strategy(
    *,
    name: str,
    rng: random.Random,
    universe: Sequence[Candidate],
    score_by_hash: Dict[str, float],
    exploration: float,
) -> Strategy:
    if name == "random":
        return RandomStrategy(rng, universe)
    if name == "greedy-known":
        return GreedyKnownStrategy(rng, universe, score_by_hash)
    if name == "ucb-taskkey":
        return UCBByTaskKeyStrategy(rng, universe, exploration=exploration)
    raise ValueError(f"Unknown strategy {name!r}")


def _simulate(
    *,
    strategy: Strategy,
    score_by_hash: Dict[str, float],
    budget: int,
    miss_policy: str,
    max_resample: int,
) -> List[StepLog]:
    """
    miss_policy:
      - "count": a miss consumes budget and is recorded as hit=False
      - "resample": keep resampling (up to max_resample) within the same step until hit, else count miss
    """
    logs: List[StepLog] = []
    best: Optional[float] = None
    misses = 0

    for step in range(1, budget + 1):
        proposed = strategy.propose()

        score = score_by_hash.get(proposed)
        hit = score is not None

        if (not hit) and miss_policy == "resample":
            tries = 0
            while tries < max_resample and not hit:
                tries += 1
                proposed = strategy.propose()
                score = score_by_hash.get(proposed)
                hit = score is not None

        if not hit:
            misses += 1
            strategy.observe(proposed, None)
        else:
            strategy.observe(proposed, score)
            best = score if best is None else max(best, score)

        logs.append(
            StepLog(
                step=step,
                proposed_hash=proposed,
                hit=hit,
                score=score,
                best_score=best,
                misses_so_far=misses,
            )
        )

    return logs


def _summarize(logs: Sequence[StepLog]) -> Dict[str, Any]:
    hits = sum(1 for x in logs if x.hit)
    misses = len(logs) - hits
    best = next((x.best_score for x in reversed(logs) if x.best_score is not None), None)
    return {
        "budget": len(logs),
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / len(logs) if logs else 0.0,
        "best_score": best,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline search strategy analysis on cached eval results.")
    p.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results.json",
        help="Path to results.json written by experiment.py (default: experiments/param-opti/results.json)",
    )
    p.add_argument(
        "--strategy",
        choices=["random", "ucb-taskkey", "greedy-known", "kgpipe_random", "kgpipe_qgns", "kgpipe_hnr", "kgpipe_bayes"],
        default="random",
        help="Search strategy to simulate (greedy-known is an upper-bound baseline).",
    )
    p.add_argument(
        "--miss-score",
        type=float,
        default=0.5,
        help="When using kgpipe_* strategies, score to return for cache misses.",
    )
    p.add_argument(
        "--init-budget",
        type=int,
        default=3,
        help="Initialization budget for kgpipe_qgns/kgpipe_hnr/kgpipe_bayes.",
    )
    p.add_argument(
        "--init-strategy",
        choices=["random", "implementation_aware"],
        default="implementation_aware",
        help="Initialization strategy for kgpipe_* strategies.",
    )
    p.add_argument("--k", type=int, default=3, help="Top-k anchors for kgpipe_qgns.")
    p.add_argument("--rho", type=float, default=0.2, help="Exploration probability for kgpipe_qgns/kgpipe_hnr.")
    p.add_argument("--pool-size", type=int, default=32, help="Candidate pool size for kgpipe_bayes.")
    p.add_argument("--beta", type=float, default=0.5, help="Acquisition beta for kgpipe_bayes.")
    p.add_argument("--budget", type=int, default=50, help="Number of proposals to simulate.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    p.add_argument(
        "--miss-policy",
        choices=["count", "resample"],
        default="count",
        help="How to handle proposing configs without cached score.",
    )
    p.add_argument(
        "--max-resample",
        type=int,
        default=50,
        help="When miss-policy=resample, max resamples per step.",
    )
    p.add_argument(
        "--include-missing-snapshots",
        action="store_true",
        help="Include entries even if config_path snapshot file is missing.",
    )
    p.add_argument(
        "--exploration",
        type=float,
        default=2.0,
        help="Exploration coefficient for ucb-taskkey.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report with step logs.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results_path: Path = args.results
    if not results_path.exists():
        raise SystemExit(f"results.json not found: {results_path}")

    score_by_hash, entry_by_hash = _load_cache(results_path)

    rng = random.Random(args.seed)

    if str(args.strategy).startswith("kgpipe_"):
        if not _HAS_KGPIPE_SEARCH:
            raise SystemExit(
                "kgpipe_search imports failed in this environment. "
                "Run within the project environment where kgpipe_search is importable. "
                f"Root cause: {type(_KGPIPE_SEARCH_IMPORT_ERROR).__name__}: {_KGPIPE_SEARCH_IMPORT_ERROR}"
            )
        score_by_key = _load_score_by_snapshot_key(
            results_path=results_path,
            entry_by_hash=entry_by_hash,
            score_by_hash=score_by_hash,
        )
        if not score_by_key:
            raise SystemExit(
                "No cached snapshots could be loaded to score PipelineConfig objects. "
                "This usually means the `config_path` files referenced by results.json are missing. "
                "Either re-run experiment.py with an output-dir you keep, or point --results at a file "
                "whose config_path entries exist."
            )
        oracle = OfflineCacheOracle(score_by_key, miss_score=float(args.miss_score))

        if args.strategy == "kgpipe_random":
            run = random_search(
                budget=int(args.budget),
                evaluate_fn=oracle.evaluate,
                search_space=RDF_SEARCH_SPACE,
                pipeline_layout=RDF_PIPELINE_LAYOUT,
                rng=rng,
            )
        elif args.strategy == "kgpipe_qgns":
            run = qgns_search(
                budget=int(args.budget),
                init_budget=int(args.init_budget),
                init_strategy=str(args.init_strategy),
                y=1,
                evaluate_fn=oracle.evaluate,
                search_space=RDF_SEARCH_SPACE,
                pipeline_layout=RDF_PIPELINE_LAYOUT,
                k=int(args.k),
                rho=float(args.rho),
                rng=rng,
            )
        elif args.strategy == "kgpipe_hnr":
            run = hnr_search(
                budget=int(args.budget),
                init_budget=int(args.init_budget),
                init_strategy=str(args.init_strategy),
                y=1,
                evaluate_fn=oracle.evaluate,
                search_space=RDF_SEARCH_SPACE,
                pipeline_layout=RDF_PIPELINE_LAYOUT,
                rho=float(args.rho),
                rng=rng,
            )
        elif args.strategy == "kgpipe_bayes":
            run = bayesian_optimization(
                budget=int(args.budget),
                init_random=int(args.init_budget),
                init_strategy=str(args.init_strategy),
                y=1,
                evaluate_fn=oracle.evaluate,
                search_space=RDF_SEARCH_SPACE,
                pipeline_layout=RDF_PIPELINE_LAYOUT,
                pool_size=int(args.pool_size),
                beta=float(args.beta),
                rng=rng,
            )
        else:
            raise SystemExit(f"Unknown kgpipe strategy {args.strategy!r}")

        best = max((s for s, _cfg in run.history), default=None)
        print(f"results: {results_path}")
        print(f"strategy: {args.strategy}")
        print(
            f"budget: {run.budget}  cache_hit_rate: {oracle.hits / max(1, oracle.hits + oracle.misses):.3f}  best_score: {best}"
        )

        if args.out is not None:
            report = {
                "results_path": str(results_path),
                "strategy": args.strategy,
                "seed": args.seed,
                "budget": args.budget,
                "miss_score": args.miss_score,
                "cache": {"hits": oracle.hits, "misses": oracle.misses},
                "best_score": best,
                "decisions": run.decisions,
                "history": [
                    {"score": float(score), "snapshot_key": pipeline_config_snapshot_key(cfg, RDF_SEARCH_SPACE)}
                    for score, cfg in run.history
                ],
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote: {args.out}")
    else:
        candidates = _load_candidates(
            results_path=results_path,
            entry_by_hash=entry_by_hash,
            include_missing_snapshots=bool(args.include_missing_snapshots),
        )
        if not candidates:
            raise SystemExit("No candidates found (check results.json and config_path files).")
        strategy = _build_strategy(
            name=args.strategy,
            rng=rng,
            universe=candidates,
            score_by_hash=score_by_hash,
            exploration=float(args.exploration),
        )

        logs = _simulate(
            strategy=strategy,
            score_by_hash=score_by_hash,
            budget=int(args.budget),
            miss_policy=str(args.miss_policy),
            max_resample=int(args.max_resample),
        )
        summary = _summarize(logs)

        print(f"results: {results_path}")
        print(f"strategy: {args.strategy}")
        print(
            f"budget: {summary['budget']}  hit_rate: {summary['hit_rate']:.3f}  best_score: {summary['best_score']}"
        )

    if args.out is not None:
        # Note: kgpipe_* branch handles writing its own report earlier.
        if not str(args.strategy).startswith("kgpipe_"):
            report = {
                "results_path": str(results_path),
                "strategy": args.strategy,
                "seed": args.seed,
                "budget": args.budget,
                "miss_policy": args.miss_policy,
                "max_resample": args.max_resample,
                "summary": summary,
                "steps": [x._asdict() for x in logs],
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
