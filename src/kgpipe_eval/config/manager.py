from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml
from pydantic import BaseModel

from kgpipe.common import KG
from kgpipe.common.model.data import DataFormat

from kgpipe_eval.metrics.duplicates import DuplicateConfig
from kgpipe_eval.metrics.triple_alignment import TripleAlignmentConfig
from kgpipe_eval.metrics.consistency_violations import ConsistencyViolationsConfig
from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig


MetricConfigModel = BaseModel


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge override into base recursively (override wins).
    """
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], Mapping)
            and isinstance(v, Mapping)
        ):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _kg_from_path(path: Path, *, name: str | None = None) -> KG:
    """
    Build a minimal `kgpipe.common.KG` from a filesystem path.

    Notes:
    - We infer `format` from the file suffix when possible, otherwise fall back to JSON.
    - The KG object lazily parses the graph when `get_graph()` is called.
    """
    suffix = path.suffix.lower().lstrip(".")
    try:
        fmt = DataFormat(suffix)
    except Exception:
        fmt = DataFormat.JSON

    return KG(
        id=str(path),
        name=(name or path.stem),
        path=path,
        format=fmt,
    )


def _resolve_entity_alignment_config(
    metric_cfg: Mapping[str, Any],
    named: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Resolve an entity alignment config from either:
    - inline: `entity_alignment_config: {...}`
    - ref: `entity_alignment_config_ref: name`
    Optionally supports both; inline values override the referenced dict.
    """
    inline = metric_cfg.get("entity_alignment_config") or {}
    ref_name = metric_cfg.get("entity_alignment_config_ref")
    if ref_name is None:
        if not isinstance(inline, Mapping):
            raise TypeError("`entity_alignment_config` must be a mapping if provided.")
        return dict(inline)

    if not isinstance(ref_name, str) or not ref_name:
        raise TypeError("`entity_alignment_config_ref` must be a non-empty string.")
    if ref_name not in named:
        raise KeyError(f"Unknown entity alignment config ref: {ref_name!r}")

    if not isinstance(inline, Mapping):
        raise TypeError("`entity_alignment_config` must be a mapping if provided.")
    return _deep_merge_dict(named[ref_name], inline)


def load_metric_configs(config_path: str | Path) -> dict[str, MetricConfigModel]:
    """
    Load a single YAML file that defines metric configs and optional shared sub-configs.

    Expected YAML structure (minimal):

    ```yaml
    entity_alignment_configs:
      default:
        method: label_embedding
        verified_entities_path: path/to/entities.csv
        entity_sim_threshold: 0.95

    metrics:
      entity_align:
        entity_alignment_config_ref: default

      duplicates:
        entity_alignment_config_ref: default

      triple_alignment:
        reference_kg_path: path/to/reference.nt
        entity_alignment_config_ref: default
        value_sim_threshold: 0.5
    ```

    Returned dict keys are metric keys (e.g. "duplicates") and values are instantiated
    Pydantic config objects (e.g. `DuplicateConfig`).
    """
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("Top-level YAML must be a mapping/dict.")

    named_entity_alignment: dict[str, dict[str, Any]] = {}
    raw_named = raw.get("entity_alignment_configs") or {}
    if raw_named:
        if not isinstance(raw_named, Mapping):
            raise TypeError("`entity_alignment_configs` must be a mapping/dict.")
        for k, v in raw_named.items():
            if not isinstance(k, str) or not k:
                raise TypeError("`entity_alignment_configs` keys must be non-empty strings.")
            if not isinstance(v, Mapping):
                raise TypeError(f"`entity_alignment_configs.{k}` must be a mapping/dict.")
            named_entity_alignment[k] = dict(v)

    metrics_raw = raw.get("metrics") or {}
    if not isinstance(metrics_raw, Mapping):
        raise TypeError("`metrics` must be a mapping/dict.")

    out: dict[str, MetricConfigModel] = {}
    for metric_key, metric_cfg_any in metrics_raw.items():
        if not isinstance(metric_key, str) or not metric_key:
            raise TypeError("Metric keys in `metrics` must be non-empty strings.")
        if metric_cfg_any is None:
            metric_cfg: dict[str, Any] = {}
        elif isinstance(metric_cfg_any, Mapping):
            metric_cfg = dict(metric_cfg_any)
        else:
            raise TypeError(f"`metrics.{metric_key}` must be a mapping/dict.")

        # --- Metric-specific instantiation rules
        if metric_key in {"entity_align", "entity_alignment"}:
            entity_cfg_dict = _resolve_entity_alignment_config(metric_cfg, named_entity_alignment)
            # Allow `reference_kg_path` convenience here too
            if "reference_kg_path" in entity_cfg_dict and "reference_kg" not in entity_cfg_dict:
                ref_path = Path(entity_cfg_dict.pop("reference_kg_path"))
                entity_cfg_dict["reference_kg"] = _kg_from_path(ref_path)
            out[metric_key] = EntityAlignmentConfig.model_validate(entity_cfg_dict)
            continue

        if metric_key in {"duplicates", "duplicate"}:
            entity_cfg_dict = _resolve_entity_alignment_config(metric_cfg, named_entity_alignment)
            if "reference_kg_path" in entity_cfg_dict and "reference_kg" not in entity_cfg_dict:
                ref_path = Path(entity_cfg_dict.pop("reference_kg_path"))
                entity_cfg_dict["reference_kg"] = _kg_from_path(ref_path)
            out[metric_key] = DuplicateConfig.model_validate(
                {
                    "entity_alignment_config": EntityAlignmentConfig.model_validate(entity_cfg_dict),
                }
            )
            continue

        if metric_key in {"triple_alignment", "triple_align"}:
            cfg_dict: dict[str, Any] = dict(metric_cfg)
            entity_cfg_dict = _resolve_entity_alignment_config(metric_cfg, named_entity_alignment)
            if "reference_kg_path" in entity_cfg_dict and "reference_kg" not in entity_cfg_dict:
                ref_path = Path(entity_cfg_dict.pop("reference_kg_path"))
                entity_cfg_dict["reference_kg"] = _kg_from_path(ref_path)
            cfg_dict["entity_alignment_config"] = EntityAlignmentConfig.model_validate(entity_cfg_dict)

            # Allow YAML to specify a path rather than an in-memory KG object
            if "reference_kg_path" in cfg_dict and "reference_kg" not in cfg_dict:
                ref_path = Path(cfg_dict.pop("reference_kg_path"))
                cfg_dict["reference_kg"] = _kg_from_path(ref_path)

            out[metric_key] = TripleAlignmentConfig.model_validate(cfg_dict)
            continue

        if metric_key in {
            "consistency_violations",
            "disjoint_domain",
            "domain",
            "range",
            "relation_direction",
            "datatype",
            "datatype_format",
        }:
            cfg_dict = dict(metric_cfg)
            if "reference_kg_path" in cfg_dict and "reference_kg" not in cfg_dict:
                ref_path = Path(cfg_dict.pop("reference_kg_path"))
                cfg_dict["reference_kg"] = _kg_from_path(ref_path)
            out[metric_key] = ConsistencyViolationsConfig.model_validate(cfg_dict)
            continue

        raise KeyError(
            f"Unknown metric key {metric_key!r} in config. "
            "Add it to `kgpipe_eval.config.manager.load_metric_configs`."
        )

    return out

