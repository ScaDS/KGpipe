from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .models import PipelineConfig, PipelineFamily
from .pipeline_util import (
    compute_rdf_metrics,
    compute_te_metrics,
    default_base_workdir,
    run_pipeline_for_config,
    TEST_DATA_ONTOLOGY_PATH,
    # _test_data_path,
)


@dataclass(frozen=True)
class QualityBreakdown:
    accuracy: float
    coverage: float
    consistency: float
    total: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _base_quality_components(cfg: PipelineConfig) -> tuple[float, float, float]:
    """
    Deterministic (noise-free) quality components for a configuration.

    This is used both for the simulated "true" evaluation (with added noise)
    and for the approximate estimator (with different noise).
    """
    if cfg.family == PipelineFamily.RDF:
        impl_acc = 0.0
        impl_cov = 0.0
        impl_con = 0.0

        om = cfg.implementations["ontology_matching"]
        if om == "string_sim":
            impl_con += 0.01
        elif om == "embedding_sim":
            impl_acc += 0.05
            impl_cov += 0.02
        elif om == "hybrid":
            impl_acc += 0.06
            impl_cov += 0.03
            impl_con += 0.01
        elif om == "llm_alignment":
            impl_cov += 0.05
            impl_acc += 0.04
            impl_con -= 0.01

        em = cfg.implementations["entity_matching"]
        if em == "rule_based":
            impl_con += 0.02
        elif em == "blocking_sim":
            impl_acc += 0.04
            impl_cov += 0.02
        elif em == "embedding_er":
            impl_acc += 0.06
            impl_cov += 0.03
        elif em == "llm_er":
            impl_cov += 0.05
            impl_acc += 0.05
            impl_con -= 0.01

        fu = cfg.implementations["fusion"]
        if fu == "union":
            impl_cov += 0.03
        elif fu == "majority_vote":
            impl_con += 0.03
            impl_acc += 0.01
        elif fu == "quality_weighted":
            impl_con += 0.06
            impl_acc += 0.02

        s_thr = float(cfg.params["schema_sim_threshold"])
        e_thr = float(cfg.params["entity_sim_threshold"])
        f_thr = float(cfg.params["fusion_confidence_threshold"])
        bk = float(cfg.params.get("blocking_key_strength", 0.5))

        acc = 0.55 + 0.18 * _sigmoid((s_thr - 0.65) * 8) + 0.18 * _sigmoid((e_thr - 0.65) * 8)
        cov = 0.65 - 0.25 * _sigmoid((s_thr - 0.6) * 7) - 0.25 * _sigmoid((e_thr - 0.6) * 7)
        con = 0.55 + 0.20 * _sigmoid((f_thr - 0.45) * 6)

        strict = (s_thr + e_thr) / 2.0
        con -= 0.05 * _sigmoid((strict - 0.85) * 10)

        cov += 0.03 * _sigmoid((bk - 0.3) * 6)
        acc -= 0.02 * _sigmoid((bk - 0.8) * 10)

        acc += impl_acc
        cov += impl_cov
        con += impl_con

        return acc, cov, con

    if cfg.family == PipelineFamily.TEXT:
        impl_acc = 0.0
        impl_cov = 0.0
        impl_con = 0.0

        ie = cfg.implementations["information_extraction"]
        if ie == "pattern_ie":
            impl_con += 0.01
        elif ie == "openie":
            impl_cov += 0.04
            impl_acc += 0.01
        elif ie == "hybrid_ie":
            impl_cov += 0.06
            impl_acc += 0.02
            impl_con += 0.01
        elif ie == "llm_ie":
            impl_cov += 0.08
            impl_acc += 0.03
            impl_con -= 0.01

        el = cfg.implementations["entity_linking"]
        if el == "dictionary_linking":
            impl_cov += 0.02
        elif el == "embedding_linking":
            impl_acc += 0.06
        elif el == "llm_linking":
            impl_acc += 0.07
            impl_cov += 0.02
            impl_con -= 0.01

        fu = cfg.implementations["fusion"]
        if fu == "union":
            impl_cov += 0.03
        elif fu == "majority_vote":
            impl_con += 0.03
            impl_acc += 0.01
        elif fu == "quality_weighted":
            impl_con += 0.07
            impl_acc += 0.02

        ie_thr = float(cfg.params["ie_conf_threshold"])
        link_thr = float(cfg.params["link_sim_threshold"])
        f_thr = float(cfg.params["fusion_confidence_threshold"])
        cw = float(cfg.params.get("context_window", 256.0))

        acc = 0.40 + 0.22 * _sigmoid((link_thr - 0.6) * 7) + 0.10 * _sigmoid((ie_thr - 0.55) * 6)
        cov = 0.55 - 0.28 * _sigmoid((ie_thr - 0.55) * 7) - 0.18 * _sigmoid((link_thr - 0.6) * 6)
        con = 0.45 + 0.22 * _sigmoid((f_thr - 0.45) * 6)

        noisy = (0.6 - ie_thr) + (0.6 - link_thr)
        con -= 0.10 * _sigmoid(noisy * 6)

        cov += 0.03 * _sigmoid((cw - 160.0) / 60.0)
        con -= 0.02 * _sigmoid((cw - 420.0) / 70.0)

        acc += impl_acc
        cov += impl_cov
        con += impl_con

        return acc, cov, con

    raise ValueError(f"Unknown family: {cfg.family}")


def evaluate_true_quality(rng: random.Random, cfg: PipelineConfig) -> QualityBreakdown:
    """
    Real(ish) end-to-end objective: run a KGpipe pipeline for this config and
    compute measurable proxy metrics from its outputs.

    Notes:
    - This intentionally uses bundled `kgpipe_tasks/test/test_data` inputs so
      the experiments are runnable out of the box.
    - Metrics are proxy/reference-independent signals (no gold labels yet).
    """
    base = default_base_workdir()
    run = run_pipeline_for_config(cfg=cfg, base_workdir=base, stable_files=False)

    if cfg.family == PipelineFamily.RDF:
        ontology = TEST_DATA_ONTOLOGY_PATH
        m = compute_rdf_metrics(output_nt=run.final_output.path, ontology_ttl=ontology)
    else:
        m = compute_te_metrics(te_json_path=run.final_output.path)

    acc = min(1.0, max(0.0, float(m["accuracy"])))
    cov = min(1.0, max(0.0, float(m["coverage"])))
    con = min(1.0, max(0.0, float(m["consistency"])))

    total = 0.45 * acc + 0.30 * cov + 0.25 * con
    total = min(1.0, max(0.0, total))
    return QualityBreakdown(accuracy=acc, coverage=cov, consistency=con, total=total)


def estimate_quality_from_config(rng: random.Random, cfg: PipelineConfig) -> float:
    """
    Approximate estimator Q-hat used by the quality-aware search to rank candidates
    without executing the full pipeline.

    For now this remains a cheap heuristic over the config (so the search is not
    dominated by expensive runs). The "true" objective is produced by actually
    executing the pipeline in `evaluate_true_quality`.
    """
    acc, cov, con = _base_quality_components(cfg)
    # Estimator has its own noise and slight systematic distortion.
    if cfg.family == PipelineFamily.RDF:
        acc += rng.gauss(0.0, 0.015)
        cov += rng.gauss(0.0, 0.015)
        con += rng.gauss(0.0, 0.015)
    else:
        acc += rng.gauss(0.0, 0.020)
        cov += rng.gauss(0.0, 0.020)
        con += rng.gauss(0.0, 0.020)

    acc = min(1.0, max(0.0, acc))
    cov = min(1.0, max(0.0, cov))
    con = min(1.0, max(0.0, con))
    est = 0.45 * acc + 0.30 * cov + 0.25 * con
    return min(1.0, max(0.0, est))


def estimate_quality(rng: random.Random, true_total: float, family: PipelineFamily) -> float:
    raise RuntimeError(
        "estimate_quality(true_total, family) is deprecated; "
        "use estimate_quality_from_config(rng, cfg) instead."
    )

