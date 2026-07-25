#!/usr/bin/env python3
"""Re-aggregate cached ``*.eval.json`` snapshots under alternate ranking configs.

Cached eval files store per-metric measurements (and a ``final_score`` under the
default aggregation). This script recomputes ``final_score`` for every named
aggregation in ``ranking_conf.AGGREGATION_CONFIGS`` (or a chosen subset) and
writes sorted score lists — without re-running pipelines or metrics.

It also plots sorted score curves (x = config index 1..N, y = final_score) for
each aggregation, optionally side-by-side for RDF and text.

Example::

    PYTHONPATH=src python -m kgpipe_search.reaggregate_evals \\
        --eval-dir runs/rdf runs/text \\
        --out-dir runs/score_curves
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from kgpipe_search.evaluation import score_from_cached_evaluation
from kgpipe_search.ranking_conf import AGGREGATION_CONFIGS, get_aggregation_config

# Match paper-style sizing used by plot_search_evolution_aggregate.
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

AGG_LABELS = {
    "default": "default",
    "custom": "custom",
    "flat_hmean": "flat hmean",
}

AGG_LINESTYLES = {
    "default": "-",
    "custom": "--",
    "flat_hmean": ":",
}


def _config_hash_from_eval_path(path: Path) -> str:
    name = path.name
    suffix = ".eval.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def discover_eval_files(eval_dir: Path) -> List[Path]:
    files = sorted(eval_dir.glob("*.eval.json"))
    if not files:
        raise FileNotFoundError(f"No *.eval.json files found in {eval_dir}")
    return files


def load_evaluation(path: Path) -> Optional[Mapping[str, Any]]:
    """Load a cached eval payload, or ``None`` for error / unusable snapshots."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: skip {path.name}: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        print(f"warning: skip {path.name}: expected JSON object", file=sys.stderr)
        return None
    if payload.get("status") == "error":
        print(f"warning: skip {path.name}: cached error", file=sys.stderr)
        return None
    if "subgroups" not in payload and not isinstance(payload.get("final_score"), (int, float)):
        print(f"warning: skip {path.name}: no measurements or final_score", file=sys.stderr)
        return None
    return payload


def reaggregate_eval_dir(
    eval_dir: Path,
    *,
    aggregations: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return ``aggregation ->`` sorted list of ``{config_hash, final_score, eval_path}``.

    Lists are sorted by ``final_score`` descending (ties broken by config hash).
    """
    for name in aggregations:
        get_aggregation_config(name)  # validate early

    rows_by_agg: Dict[str, List[Dict[str, Any]]] = {name: [] for name in aggregations}
    skipped = 0

    for path in discover_eval_files(eval_dir):
        evaluation = load_evaluation(path)
        if evaluation is None:
            skipped += 1
            continue
        config_hash = _config_hash_from_eval_path(path)
        scores: Dict[str, float] = {}
        try:
            for name in aggregations:
                scores[name] = float(score_from_cached_evaluation(evaluation, name))
        except Exception as exc:
            print(f"warning: skip {path.name}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        for name, score in scores.items():
            rows_by_agg[name].append(
                {
                    "config_hash": config_hash,
                    "final_score": score,
                    "eval_path": str(path),
                }
            )

    for name, rows in rows_by_agg.items():
        rows.sort(key=lambda r: (-float(r["final_score"]), str(r["config_hash"])))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank

    if skipped:
        print(f"skipped {skipped} eval file(s)", file=sys.stderr)
    return rows_by_agg


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "max": None, "min": None, "best_config_hash": None}
    return {
        "n": len(rows),
        "max": float(rows[0]["final_score"]),
        "min": float(rows[-1]["final_score"]),
        "best_config_hash": rows[0]["config_hash"],
    }


def write_outputs(
    rows_by_agg: Mapping[str, List[Dict[str, Any]]],
    *,
    eval_dir: Path,
    out_dir: Path,
    also_scores_only: bool = True,
) -> Path:
    """Write combined JSON plus per-aggregation sorted lists under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "eval_dir": str(eval_dir.resolve()),
        "summary": {name: _summary(rows) for name, rows in rows_by_agg.items()},
        "rankings": {
            name: [
                {
                    "rank": row["rank"],
                    "config_hash": row["config_hash"],
                    "final_score": row["final_score"],
                }
                for row in rows
            ]
            for name, rows in rows_by_agg.items()
        },
    }
    combined_path = out_dir / "reaggregated_scores.json"
    combined_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for name, rows in rows_by_agg.items():
        ranked_path = out_dir / f"scores_{name}.json"
        ranked_path.write_text(
            json.dumps(
                [
                    {
                        "rank": row["rank"],
                        "config_hash": row["config_hash"],
                        "final_score": row["final_score"],
                    }
                    for row in rows
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if also_scores_only:
            # Ascending score list in the same style as runs/*_final_score_dist.
            scores_asc = sorted(float(r["final_score"]) for r in rows)
            dist_path = out_dir / f"scores_{name}.final_score_dist"
            dist_path.write_text(
                "".join(f'  "final_score": {s},\n' for s in scores_asc),
                encoding="utf-8",
            )
            tsv_path = out_dir / f"scores_{name}.tsv"
            tsv_path.write_text(
                "rank\tconfig_hash\tfinal_score\n"
                + "".join(
                    f"{row['rank']}\t{row['config_hash']}\t{row['final_score']}\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

    return combined_path


def _domain_label(eval_dir: Path) -> str:
    name = eval_dir.name.lower()
    if "rdf" in name:
        return "RDF"
    if "text" in name:
        return "Text"
    return eval_dir.name


def _sorted_scores_asc(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    return sorted(float(r["final_score"]) for r in rows)


def _plot_curves_on_ax(
    ax: Any,
    rows_by_agg: Mapping[str, List[Dict[str, Any]]],
    *,
    aggregations: Sequence[str],
    y_full: bool = False,
) -> None:
    all_ys: List[float] = []
    for name in aggregations:
        rows = rows_by_agg.get(name) or []
        if not rows:
            continue
        ys = _sorted_scores_asc(rows)
        all_ys.extend(ys)
        xs = list(range(1, len(ys) + 1))
        ax.plot(
            xs,
            ys,
            linestyle=AGG_LINESTYLES.get(name, "-"),
            linewidth=1.5,
            label=f"{AGG_LABELS.get(name, name)} (n={len(ys)})",
        )
    ax.set_xlabel("Configs (sorted by score)")
    ax.set_ylabel("Final score")
    if y_full:
        ax.set_ylim(0.0, 1.0)
    elif all_ys:
        lo, hi = min(all_ys), max(all_ys)
        pad = max(0.02, 0.05 * (hi - lo) if hi > lo else 0.05)
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)


def plot_sorted_score_curves(
    rows_by_agg: Mapping[str, List[Dict[str, Any]]],
    *,
    aggregations: Sequence[str],
    out: Path,
    title: str,
    y_full: bool = False,
) -> Path:
    """Plot ascending sorted score curves for each aggregation into ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_HEIGHT_IN))
        _plot_curves_on_ax(ax, rows_by_agg, aggregations=aggregations, y_full=y_full)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out, dpi=PAPER_DPI)
        plt.close(fig)
    return out


def plot_sorted_score_curves_panel(
    panels: Sequence[Tuple[str, Mapping[str, List[Dict[str, Any]]]]],
    *,
    aggregations: Sequence[str],
    out: Path,
    title: str = "",
    y_full: bool = False,
) -> Path:
    """Side-by-side sorted score curves (e.g. RDF | Text)."""
    if not panels:
        raise ValueError("panels must be non-empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(COL_WIDTH_IN * n, COL_HEIGHT_IN),
            sharey=False,
            squeeze=False,
        )
        for ax, (panel_title, rows_by_agg) in zip(axes[0], panels):
            _plot_curves_on_ax(
                ax, rows_by_agg, aggregations=aggregations, y_full=y_full
            )
            ax.set_title(panel_title)
        if title:
            fig.suptitle(title, y=1.02)
        fig.tight_layout()
        fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def _resolve_out_dir(eval_dir: Path, out_dir: Optional[Path], *, multi: bool) -> Path:
    if out_dir is None:
        return eval_dir.parent / f"{eval_dir.name}_reaggregated"
    if multi:
        return out_dir / eval_dir.name
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    known = ", ".join(sorted(AGGREGATION_CONFIGS))
    p = argparse.ArgumentParser(
        description=(
            "Recompute final_score for cached *.eval.json files under one or more "
            "rank-aggregation configs, write sorted score lists, and plot curves."
        )
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing *.eval.json (e.g. runs/rdf runs/text)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. With one --eval-dir: used directly "
            "(default <eval-dir>_reaggregated). With several: per-domain "
            "subdirs are created under this path."
        ),
    )
    p.add_argument(
        "--aggregations",
        nargs="+",
        choices=sorted(AGGREGATION_CONFIGS),
        default=sorted(AGGREGATION_CONFIGS),
        help=f"Aggregation config names to recompute (default: all of {known})",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Print top-N scores per aggregation to stdout (0 to silence)",
    )
    p.add_argument(
        "--no-scores-only",
        action="store_true",
        help="Do not write .final_score_dist / .tsv companion files",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing sorted-score curve plots",
    )
    p.add_argument(
        "--plot-title",
        type=str,
        default="",
        help="Optional title for the combined RDF|Text panel plot",
    )
    p.add_argument(
        "--y-full",
        action="store_true",
        help="Force y-axis to [0, 1] instead of fitting each panel's score range",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    eval_dirs = [p.resolve() for p in args.eval_dir]
    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            raise SystemExit(f"--eval-dir is not a directory: {eval_dir}")

    multi = len(eval_dirs) > 1
    root_out = args.out_dir.resolve() if args.out_dir is not None else None
    if multi and root_out is None:
        # Shared parent next to the first eval dir, e.g. runs/score_curves
        root_out = eval_dirs[0].parent / "score_curves"

    panel_data: List[Tuple[str, Dict[str, List[Dict[str, Any]]]]] = []

    for eval_dir in eval_dirs:
        out_dir = _resolve_out_dir(eval_dir, root_out, multi=multi)
        rows_by_agg = reaggregate_eval_dir(eval_dir, aggregations=args.aggregations)
        combined = write_outputs(
            rows_by_agg,
            eval_dir=eval_dir,
            out_dir=out_dir,
            also_scores_only=not args.no_scores_only,
        )
        print(f"wrote {combined}")
        for name, rows in rows_by_agg.items():
            summary = _summary(rows)
            print(
                f"  {name}: n={summary['n']} max={summary['max']} "
                f"best={summary['best_config_hash']}"
            )
            if args.top > 0 and rows:
                print(f"  top {min(args.top, len(rows))} ({name}):")
                for row in rows[: args.top]:
                    print(
                        f"    {row['rank']:4d}  {row['final_score']:.10f}  {row['config_hash']}"
                    )

        domain = _domain_label(eval_dir)
        n_configs = len(next(iter(rows_by_agg.values()), []))
        panel_data.append((f"{domain} (n={n_configs})", rows_by_agg))

        if not args.no_plot:
            plot_path = plot_sorted_score_curves(
                rows_by_agg,
                aggregations=args.aggregations,
                out=out_dir / "sorted_score_curve.png",
                title=f"{domain} sorted final scores",
                y_full=args.y_full,
            )
            print(f"wrote {plot_path}")

    if not args.no_plot and len(panel_data) > 1:
        panel_out = (root_out or eval_dirs[0].parent) / "sorted_score_curves.png"
        path = plot_sorted_score_curves_panel(
            panel_data,
            aggregations=args.aggregations,
            out=panel_out,
            title=args.plot_title,
            y_full=args.y_full,
        )
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
