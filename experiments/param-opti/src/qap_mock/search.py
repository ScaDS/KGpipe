from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import PipelineConfig, PipelineFamily, SearchMethod, SearchSpaceMode
from .objectives import QualityBreakdown, estimate_quality_from_config, evaluate_true_quality
from .search_space import get_family_space, mutate_config, sample_config


@dataclass
class EvaluationRecord:
    cfg: PipelineConfig
    true: QualityBreakdown
    est_total: float

    def as_dict(self) -> dict:
        return {
            "config": self.cfg.as_dict(),
            "true": {
                "accuracy": self.true.accuracy,
                "coverage": self.true.coverage,
                "consistency": self.true.consistency,
                "total": self.true.total,
            },
            "estimated_total": self.est_total,
        }


def _eval_once(rng: random.Random, cfg: PipelineConfig) -> EvaluationRecord:
    true = evaluate_true_quality(rng, cfg)
    est = estimate_quality_from_config(rng, cfg)
    return EvaluationRecord(cfg=cfg, true=true, est_total=est)


def run_search(
    *,
    seed: int,
    family: PipelineFamily,
    method: SearchMethod,
    budget: int,
    mode: SearchSpaceMode = SearchSpaceMode.JOINT,
) -> List[EvaluationRecord]:
    rng = random.Random(seed)
    space = get_family_space(family)
    default_cfg = PipelineConfig(family=family, implementations=space.default_impl, params=space.default_params)

    records: List[EvaluationRecord] = []

    if method == SearchMethod.DEFAULT:
        records.append(_eval_once(rng, default_cfg))
        return records

    if method == SearchMethod.RANDOM:
        for _ in range(budget):
            cfg = sample_config(rng, family, mode=mode, fixed_default=default_cfg)
            records.append(_eval_once(rng, cfg))
        return records

    if method == SearchMethod.QUALITY_AWARE:
        # Simple, explainable heuristic:
        # - start from default
        # - maintain incumbent based on estimated quality (Q-hat)
        # - propose new configs by mutating incumbent (exploitation)
        # - occasional random exploration
        incumbent = default_cfg
        incumbent_est: Optional[float] = None

        for t in range(budget):
            # "Lookahead" using cheap quality estimates: generate a pool of
            # candidates, pick the one with best estimated quality, then
            # spend one "real" evaluation budget on it.
            pool_size = 12 if t < 5 else 8
            candidates: List[PipelineConfig] = []
            for _ in range(pool_size):
                explore = rng.random() < (0.35 if t < 3 else 0.20)
                if explore:
                    candidates.append(sample_config(rng, family, mode=mode, fixed_default=default_cfg))
                else:
                    candidates.append(
                        mutate_config(
                            rng,
                            incumbent,
                            mode=mode,
                            p_change_impl=0.70,
                            p_change_param=0.85,
                        )
                    )

            best_est = None
            best_cfg = None
            for c in candidates:
                est = estimate_quality_from_config(rng, c)
                if best_est is None or est > best_est:
                    best_est = est
                    best_cfg = c

            assert best_cfg is not None
            cfg = best_cfg

            rec = _eval_once(rng, cfg)
            records.append(rec)

            if incumbent_est is None or rec.est_total > incumbent_est:
                incumbent = cfg
                incumbent_est = rec.est_total

        return records

    raise ValueError(f"Unknown method: {method}")


def best_so_far_curve(records: List[EvaluationRecord]) -> List[float]:
    best = -1.0
    curve: List[float] = []
    for r in records:
        best = max(best, r.true.total)
        curve.append(best)
    return curve


def evals_to_fraction_of_final_best(curve: List[float], fraction: float) -> Optional[int]:
    if not curve:
        return None
    final_best = curve[-1]
    target = fraction * final_best
    for i, v in enumerate(curve, start=1):
        if v >= target:
            return i
    return None


def summarize_best(records: List[EvaluationRecord]) -> Tuple[float, float]:
    curve = best_so_far_curve(records)
    best = curve[-1] if curve else float("nan")
    return best, best

