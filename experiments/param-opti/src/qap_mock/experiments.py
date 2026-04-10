from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .models import PipelineFamily, SearchMethod, SearchSpaceMode
from .search import (
    best_so_far_curve,
    evals_to_fraction_of_final_best,
    run_search,
)
from .search_space import get_family_space, sample_config
from .stats import mae, mean, pearsonr, spearmanr, stdev, topk_agreement
from .objectives import evaluate_true_quality, estimate_quality_from_config


@dataclass
class Exp1Cell:
    mean_best: float
    std_best: float
    mean_evals_to_95: Optional[float]

    def as_dict(self) -> dict:
        return {
            "best_score_mean": self.mean_best,
            "best_score_std": self.std_best,
            "evals_to_95_mean": self.mean_evals_to_95,
        }


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def experiment_1_search_effectiveness(
    *,
    outdir: Path,
    budget: int = 20,
    runs: int = 5,
    base_seed: int = 7,
) -> dict:
    """
    Mirrors Section 6.3 / Table 2 narrative:
    - Compare Default, Random Search, Quality-Aware Search
    - Fixed budget B=20
    - Report best achieved score (mean ± std over 5 runs)
    - Report mean evaluations to reach 95% of each run's final best
    """
    _ensure_outdir(outdir)

    methods = [SearchMethod.DEFAULT] #, SearchMethod.RANDOM, SearchMethod.QUALITY_AWARE]
    families = [PipelineFamily.RDF, PipelineFamily.TEXT]

    table: Dict[str, Dict[str, Exp1Cell]] = {}
    raw: Dict[str, Dict[str, List[dict]]] = {}

    for fam in families:
        fam_key = fam.value
        table[fam_key] = {}
        raw[fam_key] = {}

        for m in methods:
            seeds = [base_seed + i for i in range(runs)]
            bests: List[float] = []
            evals95: List[float] = []
            raw_runs: List[dict] = []

            for i, s in enumerate(seeds):
                recs = run_search(
                    seed=10_000 * (i + 1) + s,
                    family=fam,
                    method=m,
                    budget=budget,
                    mode=SearchSpaceMode.JOINT,
                )
                curve = best_so_far_curve(recs)
                bests.append(curve[-1])
                e95 = evals_to_fraction_of_final_best(curve, 0.95)
                if e95 is not None:
                    evals95.append(float(e95))

                raw_runs.append(
                    {
                        "seed": s,
                        "curve_best_so_far": curve,
                    }
                )

            cell = Exp1Cell(
                mean_best=mean(bests),
                std_best=stdev(bests) if m != SearchMethod.DEFAULT else float("nan"),
                mean_evals_to_95=mean(evals95) if (m != SearchMethod.DEFAULT and evals95) else None,
            )
            table[fam_key][m.value] = cell
            raw[fam_key][m.value] = raw_runs

    result = {
        "budget": budget,
        "runs": runs,
        "table": {
            fam: {meth: cell.as_dict() for meth, cell in methods_.items()}
            for fam, methods_ in table.items()
        },
        "raw": raw,
    }

    (outdir / "exp1_search_effectiveness.json").write_text(json.dumps(result, indent=2))
    return result


def experiment_2_estimation_reliability(
    *,
    outdir: Path,
    n_samples: int = 60,
    seed: int = 23,
    topk: int = 10,
) -> dict:
    """
    Mirrors Section 6.4 narrative:
    - sample configurations
    - compute estimated vs true scores
    - compute correlation (Pearson/Spearman), MAE, top-k agreement
    """
    _ensure_outdir(outdir)

    rng = random.Random(seed)
    families = [PipelineFamily.RDF, PipelineFamily.TEXT]

    out: Dict[str, dict] = {"n_samples": n_samples, "topk": topk, "by_family": {}}

    for fam in families:
        true_scores: List[float] = []
        est_scores: List[float] = []

        for _ in range(n_samples):
            cfg = sample_config(rng, fam, mode=SearchSpaceMode.JOINT)
            true = evaluate_true_quality(rng, cfg).total
            est = estimate_quality_from_config(rng, cfg)
            true_scores.append(true)
            est_scores.append(est)

        fam_key = fam.value
        out["by_family"][fam_key] = {
            "pearson": pearsonr(est_scores, true_scores),
            "spearman": spearmanr(est_scores, true_scores),
            "mae": mae(est_scores, true_scores),
            "topk_agreement": topk_agreement(est_scores, true_scores, topk),
        }

    (outdir / "exp2_estimation_reliability.json").write_text(json.dumps(out, indent=2))
    return out


def experiment_3_dimension_impact(
    *,
    outdir: Path,
    budget: int = 20,
    runs: int = 5,
    base_seed: int = 101,
) -> dict:
    """
    Mirrors Section 6.5 narrative:
    Compare best scores for restricted spaces:
    - implementation-only
    - parameter-only
    - joint
    """
    _ensure_outdir(outdir)

    families = [PipelineFamily.RDF, PipelineFamily.TEXT]
    modes = [
        SearchSpaceMode.IMPLEMENTATION_ONLY,
        SearchSpaceMode.PARAMETER_ONLY,
        SearchSpaceMode.JOINT,
    ]

    out: Dict[str, dict] = {"budget": budget, "runs": runs, "by_family": {}}

    for fam in families:
        fam_out: Dict[str, dict] = {}
        for mode in modes:
            bests: List[float] = []
            for i in range(runs):
                seed = base_seed + i * 17
                recs = run_search(
                    seed=20_000 * (i + 1) + seed,
                    family=fam,
                    method=SearchMethod.QUALITY_AWARE,
                    budget=budget,
                    mode=mode,
                )
                curve = best_so_far_curve(recs)
                bests.append(curve[-1])

            fam_out[mode.value] = {"best_mean": mean(bests), "best_std": stdev(bests)}

        out["by_family"][fam.value] = fam_out

    (outdir / "exp3_dimension_impact.json").write_text(json.dumps(out, indent=2))
    return out

