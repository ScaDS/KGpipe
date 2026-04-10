from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import PipelineConfig, PipelineFamily, SearchSpaceMode


@dataclass(frozen=True)
class FamilySpace:
    tasks: List[str]
    impl_choices: Dict[str, List[str]]
    param_ranges: Dict[str, Tuple[float, float]]
    default_impl: Dict[str, str]
    default_params: Dict[str, float]


def get_family_space(family: PipelineFamily) -> FamilySpace:
    # Compact but expressive, mirroring the paper text:
    # - discrete implementation choices per task
    # - continuous thresholds
    if family == PipelineFamily.RDF:
        tasks = ["ontology_matching", "entity_matching", "fusion"]
        impl_choices = {
            "ontology_matching": ["string_sim", "embedding_sim", "hybrid", "llm_alignment"],
            "entity_matching": ["rule_based", "blocking_sim", "embedding_er", "llm_er"],
            "fusion": ["union", "quality_weighted", "majority_vote"],
        }
        param_ranges = {
            "schema_sim_threshold": (0.3, 0.95),
            "entity_sim_threshold": (0.3, 0.95),
            "fusion_confidence_threshold": (0.1, 0.9),
            "blocking_key_strength": (0.0, 1.0),
        }
        default_impl = {
            "ontology_matching": "string_sim",
            "entity_matching": "rule_based",
            "fusion": "union",
        }
        default_params = {
            "schema_sim_threshold": 0.7,
            "entity_sim_threshold": 0.7,
            "fusion_confidence_threshold": 0.5,
            "blocking_key_strength": 0.5,
        }
        return FamilySpace(tasks, impl_choices, param_ranges, default_impl, default_params)

    if family == PipelineFamily.TEXT:
        tasks = ["information_extraction", "entity_linking", "fusion"]
        impl_choices = {
            "information_extraction": ["pattern_ie", "openie", "hybrid_ie", "llm_ie"],
            "entity_linking": ["dictionary_linking", "embedding_linking", "llm_linking"],
            "fusion": ["union", "quality_weighted", "majority_vote"],
        }
        param_ranges = {
            "ie_conf_threshold": (0.2, 0.95),
            "link_sim_threshold": (0.2, 0.95),
            "fusion_confidence_threshold": (0.1, 0.9),
            "context_window": (64.0, 512.0),
        }
        default_impl = {
            "information_extraction": "pattern_ie",
            "entity_linking": "dictionary_linking",
            "fusion": "union",
        }
        default_params = {
            "ie_conf_threshold": 0.6,
            "link_sim_threshold": 0.6,
            "fusion_confidence_threshold": 0.5,
            "context_window": 256.0,
        }
        return FamilySpace(tasks, impl_choices, param_ranges, default_impl, default_params)

    raise ValueError(f"Unknown family: {family}")


def sample_config(
    rng: random.Random,
    family: PipelineFamily,
    mode: SearchSpaceMode = SearchSpaceMode.JOINT,
    fixed_default: PipelineConfig | None = None,
) -> PipelineConfig:
    space = get_family_space(family)

    impl: Dict[str, str] = {}
    params: Dict[str, float] = {}

    if fixed_default is None:
        fixed_default = PipelineConfig(family=family, implementations=space.default_impl, params=space.default_params)

    if mode in (SearchSpaceMode.JOINT, SearchSpaceMode.IMPLEMENTATION_ONLY):
        for t in space.tasks:
            impl[t] = rng.choice(space.impl_choices[t])
    else:
        impl = dict(fixed_default.implementations)

    if mode in (SearchSpaceMode.JOINT, SearchSpaceMode.PARAMETER_ONLY):
        for p, (lo, hi) in space.param_ranges.items():
            params[p] = rng.uniform(lo, hi)
    else:
        params = dict(fixed_default.params)

    return PipelineConfig(family=family, implementations=impl, params=params)


def mutate_config(
    rng: random.Random,
    cfg: PipelineConfig,
    mode: SearchSpaceMode = SearchSpaceMode.JOINT,
    p_change_impl: float = 0.35,
    p_change_param: float = 0.8,
) -> PipelineConfig:
    space = get_family_space(cfg.family)
    impl = dict(cfg.implementations)
    params = dict(cfg.params)

    if mode in (SearchSpaceMode.JOINT, SearchSpaceMode.IMPLEMENTATION_ONLY) and rng.random() < p_change_impl:
        t = rng.choice(space.tasks)
        choices = [c for c in space.impl_choices[t] if c != impl[t]]
        if choices:
            impl[t] = rng.choice(choices)
        # Occasionally flip a second task implementation to escape local optima.
        if rng.random() < 0.25:
            t2 = rng.choice([x for x in space.tasks if x != t])
            choices2 = [c for c in space.impl_choices[t2] if c != impl[t2]]
            if choices2:
                impl[t2] = rng.choice(choices2)

    if mode in (SearchSpaceMode.JOINT, SearchSpaceMode.PARAMETER_ONLY) and rng.random() < p_change_param:
        p = rng.choice(list(space.param_ranges.keys()))
        lo, hi = space.param_ranges[p]
        # Gaussian step with clipping keeps changes local.
        step = rng.gauss(0.0, (hi - lo) * 0.08)
        params[p] = min(hi, max(lo, params[p] + step))
        if rng.random() < 0.25:
            p2 = rng.choice([x for x in space.param_ranges.keys() if x != p])
            lo2, hi2 = space.param_ranges[p2]
            step2 = rng.gauss(0.0, (hi2 - lo2) * 0.06)
            params[p2] = min(hi2, max(lo2, params[p2] + step2))

    return PipelineConfig(family=cfg.family, implementations=impl, params=params)

