#!/usr/bin/env python3
"""Plot search evolution (iteration vs quality score) from search result reports."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "search-results"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "search-results"

PLOT_FILENAME = "search-evolution.png"
PLOT_CHRONOLOGICAL_FILENAME = "search-evolution-chronological.png"
TABLE_CSV_FILENAME = "search-evolution-table.csv"
TABLE_MD_FILENAME = "search-evolution-table.md"

STRATEGY_LABELS = {
    "bayes-offline.json": "Bayesian optimization",
    "bayesian-results.json": "Bayesian optimization",
    "hnr-offline.json": "HNR-1",
    "hnr-results.json": "HNR-1",
    "hnr_2-offline.json": "HNR-2",
    "hnr_2-results.json": "HNR-2",
    "qgns-offline.json": "RNS",
    "qgns-results.json": "RNS",
    "random-implementation-aware-offline.json": "Random (implementation-aware)",
    "implementation-aware-results.json": "Implementation-aware",
    "random-random-offline.json": "Random",
    "random-results.json": "Random",
}

STRATEGY_NAME_LABELS = {
    "bayesian": "Bayesian optimization",
    "kgpipe_bayes": "Bayesian optimization",
    "hnr": "HNR",
    "kgpipe_hnr": "HNR",
    "qgns": "QGNS",
    "kgpipe_qgns": "QGNS",
    "implementation_aware": "Implementation-aware",
    "random": "Random",
    "kgpipe_random": "Random",
}


def _read_report(path: Path) -> dict[str, Any]:
    return _normalize_report(json.loads(path.read_text(encoding="utf-8")))


def _normalize_report(raw: dict[str, Any]) -> dict[str, Any]:
    """Adapt experiment.py results.json to the offline analyse report shape."""
    if "search_history" not in raw:
        return raw

    search = raw.get("search")
    search_dict = search if isinstance(search, dict) else {}
    history = [
        {"score": float(item["score"])}
        for item in raw["search_history"]
        if isinstance(item, dict) and "score" in item
    ]
    return {
        **raw,
        "history": history,
        "decisions": search_dict.get("decisions", []),
        "strategy": search_dict.get("strategy"),
        "init_budget": search_dict.get("init_budget"),
    }


def _init_budget(report: dict[str, Any]) -> int:
    explicit = report.get("init_budget")
    if explicit is not None:
        return int(explicit)

    search = report.get("search")
    if isinstance(search, dict) and search.get("init_budget") is not None:
        return int(search["init_budget"])

    decisions = report.get("decisions") or []
    if isinstance(decisions, list):
        return sum(1 for d in decisions if str(d).startswith("init("))
    return 0


def _discover_report_paths(results_dir: Path) -> List[Path]:
    offline = sorted(results_dir.glob("*-offline.json"))
    if offline:
        return offline
    return sorted(results_dir.glob("*-results.json"))


def _running_best(scores: Sequence[float]) -> List[float]:
    best: float | None = None
    out: List[float] = []
    for score in scores:
        best = score if best is None else max(best, score)
        out.append(best)
    return out


def _evolution_curve(
    scores: Sequence[float],
    *,
    init_budget: int,
    reorder_init: bool,
    running_best: bool = True,
) -> Tuple[List[int], List[float]]:
    if init_budget <= 0 or init_budget >= len(scores):
        xs = list(range(1, len(scores) + 1))
        ys = _running_best(scores) if running_best else list(scores)
        return xs, ys

    init_scores = list(scores[:init_budget])
    search_scores = list(scores[init_budget:])

    if reorder_init:
        init_scores = sorted(init_scores)

    ordered_scores = init_scores + search_scores
    xs = list(range(1, len(ordered_scores) + 1))
    ys = _running_best(ordered_scores) if running_best else ordered_scores
    return xs, ys


@dataclass(frozen=True)
class StrategyMetrics:
    strategy: str
    q_best: float
    evals_to_95pct: Optional[int]
    evals_to_best: Optional[int]
    aoc: float


def _area_under_curve(xs: Sequence[int], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(len(xs) - 1):
        dx = float(xs[i + 1] - xs[i])
        area += dx * (ys[i] + ys[i + 1]) / 2.0
    return area


def _evals_to_fraction(xs: Sequence[int], ys: Sequence[float], *, fraction: float) -> Optional[int]:
    if not ys:
        return None
    q_best = max(ys)
    threshold = fraction * q_best
    for x, y in zip(xs, ys):
        if y >= threshold:
            return int(x)
    return None


def _metrics_for_report(
    path: Path,
    report: dict[str, Any],
    *,
    reorder_init: bool,
    target_fraction: float,
) -> Optional[StrategyMetrics]:
    history = report.get("history")
    if not isinstance(history, list) or not history:
        return None

    scores = [float(item["score"]) for item in history if isinstance(item, dict) and "score" in item]
    if not scores:
        return None

    init_budget = _init_budget(report)
    xs, ys = _evolution_curve(
        scores,
        init_budget=init_budget,
        reorder_init=reorder_init,
        running_best=True,
    )

    return StrategyMetrics(
        strategy=_label_for(path, report),
        q_best=max(ys),
        evals_to_95pct=_evals_to_fraction(xs, ys, fraction=target_fraction),
        evals_to_best=_evals_to_fraction(xs, ys, fraction=1.0),
        aoc=_area_under_curve(xs, ys),
    )


def _format_metrics_table(rows: Sequence[StrategyMetrics]) -> List[List[str]]:
    header = ["Strategy", "Q best", "Evals to 95%", "Evals to best", "AOC"]
    body = [
        [
            row.strategy,
            f"{row.q_best:.4f}",
            str(row.evals_to_95pct) if row.evals_to_95pct is not None else "—",
            str(row.evals_to_best) if row.evals_to_best is not None else "—",
            f"{row.aoc:.2f}",
        ]
        for row in rows
    ]
    return [header, *body]


def _print_metrics_table(rows: Sequence[StrategyMetrics]) -> None:
    table = _format_metrics_table(rows)
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    for row_idx, row in enumerate(table):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if row_idx == 0:
            print("  ".join("-" * widths[i] for i in range(len(widths))))


def _write_metrics_csv(path: Path, rows: Sequence[StrategyMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "q_best", "evals_to_95pct", "evals_to_best", "aoc"])
        for row in rows:
            writer.writerow(
                [
                    row.strategy,
                    f"{row.q_best:.6f}",
                    row.evals_to_95pct,
                    row.evals_to_best,
                    f"{row.aoc:.4f}",
                ]
            )


def _write_metrics_markdown(path: Path, rows: Sequence[StrategyMetrics]) -> None:
    table = _format_metrics_table(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(table[0]) + " |",
        "| " + " | ".join("---" for _ in table[0]) + " |",
    ]
    for row in table[1:]:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_for(path: Path, report: dict[str, Any]) -> str:
    if path.name in STRATEGY_LABELS:
        return STRATEGY_LABELS[path.name]
    strategy = report.get("strategy")
    if isinstance(strategy, str) and strategy in STRATEGY_NAME_LABELS:
        return STRATEGY_NAME_LABELS[strategy]
    if isinstance(strategy, str):
        return strategy
    return path.stem


def plot_reports(
    reports: Iterable[Tuple[Path, dict[str, Any]]],
    *,
    reorder_init: bool,
    running_best: bool,
    out: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for path, report in reports:
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
            running_best=running_best,
        )
        label = _label_for(path, report)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, label=label)

        if init_budget > 0:
            ax.axvline(init_budget + 0.5, color="0.75", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best quality score so far" if running_best else "Quality score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot search evolution from JSON result reports.")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing *-offline.json or *-results.json reports.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated figures and tables.",
    )
    p.add_argument(
        "--skip-chronological-plot",
        action="store_true",
        help="Skip writing the chronological-init plot.",
    )
    p.add_argument(
        "--title",
        default="Search evolution",
        help="Plot title.",
    )
    p.add_argument(
        "--target-fraction",
        type=float,
        default=0.95,
        help="Fraction of Q best used for the evals-to-target column (default: 0.95).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir: Path = args.results_dir
    out_dir: Path = args.out_dir
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    report_paths = _discover_report_paths(results_dir)
    if not report_paths:
        raise SystemExit(
            f"No *-offline.json or *-results.json files found in {results_dir}"
        )

    reports = [(path, _read_report(path)) for path in report_paths]
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[StrategyMetrics] = []
    for path, report in reports:
        row = _metrics_for_report(
            path,
            report,
            reorder_init=True,
            target_fraction=float(args.target_fraction),
        )
        if row is not None:
            metrics.append(row)

    plot_out = out_dir / PLOT_FILENAME
    plot_reports(
        reports,
        reorder_init=True,
        running_best=True,
        out=plot_out,
        title=str(args.title),
    )
    print(f"wrote: {plot_out}")

    if not args.skip_chronological_plot:
        chrono_out = out_dir / PLOT_CHRONOLOGICAL_FILENAME
        chrono_title = f"{args.title} (chronological scores)"
        plot_reports(
            reports,
            reorder_init=False,
            running_best=False,
            out=chrono_out,
            title=chrono_title,
        )
        print(f"wrote: {chrono_out}")

    if metrics:
        table_csv = out_dir / TABLE_CSV_FILENAME
        table_md = out_dir / TABLE_MD_FILENAME
        _write_metrics_csv(table_csv, metrics)
        _write_metrics_markdown(table_md, metrics)
        print(f"wrote: {table_csv}")
        print(f"wrote: {table_md}")
        print()
        _print_metrics_table(metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
