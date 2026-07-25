#!/usr/bin/env python3
"""Aggregate search-evolution plots/tables across RNG seed runs.

Expects a parent results directory whose subdirectories are per-seed configs,
e.g.::

    rdf-search-results/
      init_3_budget_20_seed_0/
      init_3_budget_20_seed_42/
      init_3_budget_20_seed_1337/

For each config group (everything before ``_seed_<n>``), writes a mean curve
plot with a shaded band and a table of mean ± std metrics.

The table also reports how often each strategy reaches the known expected
maximum quality score (``TEXT_EXPECTED_MAX`` / ``RDF_EXPECTED_MAX``), inferred
from whether the results directory name contains ``text`` or ``rdf``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from plot_search_evolution import (
    _discover_report_paths,
    _evolution_curve,
    _init_budget,
    _label_for,
    _metrics_for_report,
    _read_report,
    StrategyMetrics,
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "rdf-search-results"

# Known global maxima (exhaustive / reference best quality scores).
# TEXT wo seed ref default 0.8503806701 custom 0.3802856400657162
# TEXT with seed ref default 0.8503806701 custom 0.849341845483141 
TEXT_EXPECTED_MAX =0.3802856400657162 
# RDF wo seed ref default 0.8018846725409015 custom 0.7712140467593951
# RDF with seed ref default 0.9615967544 custom 0.967927789101375
RDF_EXPECTED_MAX =  0.7712140467593951 

SEED_DIR_RE = re.compile(r"^(?P<config>.+)_seed_(?P<seed>\d+)$")

PLOT_FILENAME = "search-evolution-aggregated.png"
TABLE_CSV_FILENAME = "search-evolution-aggregated-table.csv"
TABLE_MD_FILENAME = "search-evolution-aggregated-table.md"

# Absolute tolerance when comparing run Q-best to the expected max.
EXPECTED_MAX_ABS_TOL = 1e-9

# Single-column figure for double-column papers (~3.5" column width).
# Size fonts for 1:1 print (do not shrink a wide figure in LaTeX).
COL_WIDTH_IN = 3.5
COL_HEIGHT_IN = 2.6
PAPER_DPI = 300
PAPER_RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "grid.linewidth": 0.5,
}


@dataclass(frozen=True)
class RunCurve:
    strategy: str
    seed: str
    xs: List[int]
    ys: List[float]
    init_budget: int
    metrics: StrategyMetrics


@dataclass(frozen=True)
class AggregatedMetrics:
    strategy: str
    n: int
    q_best_mean: float
    q_best_std: float
    hits_expected_max: int
    expected_max: Optional[float]
    evals_to_95pct_mean: float
    evals_to_95pct_std: float
    evals_to_best_mean: float
    evals_to_best_std: float
    aoc_mean: float
    aoc_std: float


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _fmt_mean_std(mean: float, std: float, *, digits: int) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _expected_max_for_results_dir(results_dir: Path) -> Optional[float]:
    """Pick TEXT/RDF expected max from the results directory name."""
    name = results_dir.name.lower()
    if "text" in name:
        return TEXT_EXPECTED_MAX
    if "rdf" in name:
        return RDF_EXPECTED_MAX
    return None


def _reaches_expected_max(q_best: float, expected_max: float) -> bool:
    return math.isclose(q_best, expected_max, rel_tol=0.0, abs_tol=EXPECTED_MAX_ABS_TOL)


def _discover_seed_dirs(results_dir: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """Map config key -> list of (seed, path) for ``*_seed_<n>`` subdirs."""
    groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for path in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        match = SEED_DIR_RE.match(path.name)
        if not match:
            continue
        groups[match.group("config")].append((match.group("seed"), path))
    return dict(groups)


def _load_run_curves(
    run_dir: Path,
    *,
    seed: str,
    reorder_init: bool,
    target_fraction: float,
) -> List[RunCurve]:
    curves: List[RunCurve] = []
    for path in _discover_report_paths(run_dir):
        report = _read_report(path)
        metrics = _metrics_for_report(
            path,
            report,
            reorder_init=reorder_init,
            target_fraction=target_fraction,
        )
        if metrics is None:
            continue

        history = report.get("history")
        if not isinstance(history, list) or not history:
            continue
        scores = [float(item["score"]) for item in history if isinstance(item, dict) and "score" in item]
        if not scores:
            continue

        init_budget = _init_budget(report)
        xs, ys = _evolution_curve(
            scores,
            init_budget=init_budget,
            reorder_init=reorder_init,
            running_best=True,
        )
        curves.append(
            RunCurve(
                strategy=_label_for(path, report),
                seed=seed,
                xs=xs,
                ys=ys,
                init_budget=init_budget,
                metrics=metrics,
            )
        )
    return curves


def _band_bounds(
    values: Sequence[float],
    *,
    band: str,
) -> Tuple[float, float, float]:
    mean = _mean(values)
    if band == "range":
        return mean, min(values), max(values)
    if band == "std":
        s = _std(values)
        return mean, mean - s, mean + s
    if band == "sem":
        s = _std(values)
        sem = s / math.sqrt(len(values)) if values else 0.0
        return mean, mean - sem, mean + sem
    raise ValueError(f"Unknown band mode: {band}")


def _aggregate_curves(
    curves: Sequence[RunCurve],
    *,
    band: str,
) -> Tuple[List[int], List[float], List[float], List[float], int]:
    if not curves:
        return [], [], [], [], 0

    min_len = min(len(c.ys) for c in curves)
    xs = list(range(1, min_len + 1))
    means: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []
    for i in range(min_len):
        vals = [c.ys[i] for c in curves]
        mean, lo, hi = _band_bounds(vals, band=band)
        means.append(mean)
        lowers.append(lo)
        uppers.append(hi)
    return xs, means, lowers, uppers, len(curves)


def _aggregate_metrics(
    curves: Sequence[RunCurve],
    *,
    expected_max: Optional[float],
) -> AggregatedMetrics:
    q_best = [c.metrics.q_best for c in curves]
    aoc = [c.metrics.aoc for c in curves]
    to_95 = [float(c.metrics.evals_to_95pct) for c in curves if c.metrics.evals_to_95pct is not None]
    to_best = [float(c.metrics.evals_to_best) for c in curves if c.metrics.evals_to_best is not None]
    hits = (
        sum(1 for q in q_best if _reaches_expected_max(q, expected_max))
        if expected_max is not None
        else 0
    )
    return AggregatedMetrics(
        strategy=curves[0].strategy,
        n=len(curves),
        q_best_mean=_mean(q_best),
        q_best_std=_std(q_best),
        hits_expected_max=hits,
        expected_max=expected_max,
        evals_to_95pct_mean=_mean(to_95),
        evals_to_95pct_std=_std(to_95),
        evals_to_best_mean=_mean(to_best),
        evals_to_best_std=_std(to_best),
        aoc_mean=_mean(aoc),
        aoc_std=_std(aoc),
    )


def _fmt_hits(row: AggregatedMetrics) -> str:
    if row.expected_max is None:
        return "—"
    return f"{row.hits_expected_max}/{row.n}"


def _format_metrics_table(rows: Sequence[AggregatedMetrics]) -> List[List[str]]:
    header = ["Strategy", "n", "Q best", "Hits max", "Evals to 95%", "Evals to best", "AOC"]
    body = [
        [
            row.strategy,
            str(row.n),
            _fmt_mean_std(row.q_best_mean, row.q_best_std, digits=4),
            _fmt_hits(row),
            _fmt_mean_std(row.evals_to_95pct_mean, row.evals_to_95pct_std, digits=2),
            _fmt_mean_std(row.evals_to_best_mean, row.evals_to_best_std, digits=2),
            _fmt_mean_std(row.aoc_mean, row.aoc_std, digits=2),
        ]
        for row in rows
    ]
    return [header, *body]


def _print_metrics_table(rows: Sequence[AggregatedMetrics]) -> None:
    table = _format_metrics_table(rows)
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    for row_idx, row in enumerate(table):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if row_idx == 0:
            print("  ".join("-" * widths[i] for i in range(len(widths))))


def _write_metrics_csv(path: Path, rows: Sequence[AggregatedMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "strategy",
                "n",
                "q_best_mean",
                "q_best_std",
                "hits_expected_max",
                "expected_max",
                "evals_to_95pct_mean",
                "evals_to_95pct_std",
                "evals_to_best_mean",
                "evals_to_best_std",
                "aoc_mean",
                "aoc_std",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.strategy,
                    row.n,
                    f"{row.q_best_mean:.6f}",
                    f"{row.q_best_std:.6f}",
                    row.hits_expected_max,
                    f"{row.expected_max:.10f}" if row.expected_max is not None else "",
                    f"{row.evals_to_95pct_mean:.4f}",
                    f"{row.evals_to_95pct_std:.4f}",
                    f"{row.evals_to_best_mean:.4f}",
                    f"{row.evals_to_best_std:.4f}",
                    f"{row.aoc_mean:.4f}",
                    f"{row.aoc_std:.4f}",
                ]
            )


def _write_metrics_markdown(path: Path, rows: Sequence[AggregatedMetrics]) -> None:
    table = _format_metrics_table(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(table[0]) + " |",
        "| " + " | ".join("---" for _ in table[0]) + " |",
    ]
    for row in table[1:]:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_aggregated(
    by_strategy: Dict[str, List[RunCurve]],
    *,
    band: str,
    out: Path,
    title: str,
) -> None:
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_HEIGHT_IN))
        init_budget: Optional[int] = None

        for strategy in sorted(by_strategy):
            curves = by_strategy[strategy]
            xs, means, lowers, uppers, n = _aggregate_curves(curves, band=band)
            if not xs:
                continue
            if init_budget is None and curves:
                init_budget = curves[0].init_budget

            (line,) = ax.plot(
                xs,
                means,
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                label=f"{strategy} (n={n})",
            )
            ax.fill_between(xs, lowers, uppers, color=line.get_color(), alpha=0.2, linewidth=0)

        if init_budget and init_budget > 0:
            ax.axvline(init_budget + 0.5, color="0.75", linestyle=":", linewidth=0.8)

        band_label = {"range": "min–max range", "std": "±1 std", "sem": "±1 SEM"}[band]
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best quality (mean)")
        ax.set_title(f"{title}\n(shaded: {band_label})")
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="lower right",
            frameon=True,
            borderpad=0.3,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.4,
            borderaxespad=0.3,
        )
        fig.tight_layout(pad=0.35)

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight")
        plt.close(fig)


def _process_config_group(
    config: str,
    seed_dirs: Sequence[Tuple[str, Path]],
    *,
    out_dir: Path,
    title: str,
    band: str,
    target_fraction: float,
    expected_max: Optional[float],
) -> None:
    by_strategy: Dict[str, List[RunCurve]] = defaultdict(list)
    for seed, run_dir in seed_dirs:
        for curve in _load_run_curves(
            run_dir,
            seed=seed,
            reorder_init=True,
            target_fraction=target_fraction,
        ):
            by_strategy[curve.strategy].append(curve)

    if not by_strategy:
        print(f"skip {config}: no usable reports in {[p.name for _, p in seed_dirs]}")
        return

    metrics = [
        _aggregate_metrics(curves, expected_max=expected_max)
        for _, curves in sorted(by_strategy.items())
    ]
    metrics.sort(key=lambda row: row.strategy)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_out = out_dir / PLOT_FILENAME
    plot_aggregated(by_strategy, band=band, out=plot_out, title=title)
    print(f"wrote: {plot_out}")

    table_csv = out_dir / TABLE_CSV_FILENAME
    table_md = out_dir / TABLE_MD_FILENAME
    _write_metrics_csv(table_csv, metrics)
    _write_metrics_markdown(table_md, metrics)
    print(f"wrote: {table_csv}")
    print(f"wrote: {table_md}")
    print()
    expected_label = (
        f"expected_max={expected_max:.10f}" if expected_max is not None else "expected_max=none"
    )
    print(f"[{config}] seeds={[s for s, _ in seed_dirs]} {expected_label}")
    _print_metrics_table(metrics)
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate search-evolution plots/tables across seed runs."
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Parent directory containing *_seed_<n> subdirectories.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <results-dir>/<config>/ or <results-dir> if one config).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Only aggregate this config prefix (e.g. init_3_budget_20).",
    )
    p.add_argument(
        "--band",
        choices=("range", "std", "sem"),
        default="range",
        help="Shaded band around the mean curve: min-max range, ±1 std, or ±1 SEM (default: range).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Plot title prefix (default derived from config).",
    )
    p.add_argument(
        "--target-fraction",
        type=float,
        default=0.95,
        help="Fraction of Q best used for evals-to-target (default: 0.95).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    groups = _discover_seed_dirs(results_dir)
    if not groups:
        raise SystemExit(
            f"No *_seed_<n> subdirectories found in {results_dir}"
        )

    if args.config is not None:
        if args.config not in groups:
            raise SystemExit(
                f"Config {args.config!r} not found. Available: {sorted(groups)}"
            )
        groups = {args.config: groups[args.config]}

    expected_max = _expected_max_for_results_dir(results_dir)
    if expected_max is None:
        print(
            f"warning: could not infer TEXT/RDF expected max from {results_dir.name!r}; "
            "Hits max column will be empty"
        )

    for config, seed_dirs in sorted(groups.items()):
        if args.out_dir is not None:
            out_dir = args.out_dir if len(groups) == 1 else args.out_dir / config
        else:
            out_dir = results_dir / config if len(groups) > 1 else results_dir

        title = args.title or f"Search evolution ({config}, aggregated over seeds)"
        _process_config_group(
            config,
            seed_dirs,
            out_dir=out_dir,
            title=title,
            band=str(args.band),
            target_fraction=float(args.target_fraction),
            expected_max=expected_max,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
