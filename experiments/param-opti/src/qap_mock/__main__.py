from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import (
    experiment_1_search_effectiveness,
    experiment_2_estimation_reliability,
    experiment_3_dimension_impact,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Mock experiments for Quality_Aware_Pipelines.pdf (quality-aware search)"
    )
    p.add_argument(
        "which",
        choices=["exp1", "exp2", "exp3", "all"],
        help="Which experiment(s) to run",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "output_qap_mock",
        help="Output directory for JSON results",
    )
    p.add_argument("--budget", type=int, default=20, help="Evaluation budget B (exp1/exp3)")
    p.add_argument("--runs", type=int, default=5, help="Number of runs/seeds (exp1/exp3)")
    p.add_argument("--samples", type=int, default=60, help="Number of sampled configs (exp2)")

    args = p.parse_args(argv)

    results: dict[str, object] = {}

    if args.which in ("exp1", "all"):
        results["exp1"] = experiment_1_search_effectiveness(
            outdir=args.outdir, budget=args.budget, runs=args.runs
        )
    if args.which in ("exp2", "all"):
        results["exp2"] = experiment_2_estimation_reliability(
            outdir=args.outdir, n_samples=args.samples
        )
    if args.which in ("exp3", "all"):
        results["exp3"] = experiment_3_dimension_impact(
            outdir=args.outdir, budget=args.budget, runs=args.runs
        )

    # Short stdout summary so it's easy to sanity-check runs.
    print(json.dumps({"outdir": str(args.outdir), "ran": list(results.keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

