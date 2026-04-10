from __future__ import annotations

from pathlib import Path

from kgpipe_eval.config.manager import load_metric_configs, generate_default_config_dict
from kgpipe_eval.metrics.duplicates import DuplicateConfig
from kgpipe_eval.metrics.triple_alignment import TripleAlignmentConfig
from kgpipe_eval.utils.alignment_utils import EntityAlignmentConfig


def test_load_metric_configs_resolves_entity_alignment_refs(tmp_path: Path) -> None:
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        """
entity_alignment_configs:
  default:
    method: label_embedding
    verified_entities_path: tmp_test_data/verified_entities.csv
    verified_entities_delimiter: ","
    entity_sim_threshold: 0.95

metrics:
  entity_align:
    entity_alignment_config_ref: default

  duplicates:
    entity_alignment_config_ref: default

  triple_alignment:
    reference_kg_path: tmp_test_data/reference.nt
    entity_alignment_config_ref: default
    value_sim_threshold: 0.6
""".lstrip(),
        encoding="utf-8",
    )

    loaded = load_metric_configs(cfg)
    assert "entity_align" in loaded
    assert "duplicates" in loaded
    assert "triple_alignment" in loaded

    assert isinstance(loaded["entity_align"], EntityAlignmentConfig)
    assert isinstance(loaded["duplicates"], DuplicateConfig)
    assert isinstance(loaded["triple_alignment"], TripleAlignmentConfig)

    assert loaded["entity_align"].verified_entities_delimiter == ","
    assert loaded["duplicates"].entity_alignment_config.verified_entities_delimiter == ","
    assert loaded["triple_alignment"].entity_alignment_config.verified_entities_delimiter == ","

    # reference_kg is constructed from reference_kg_path
    assert loaded["triple_alignment"].reference_kg.path.as_posix().endswith("tmp_test_data/reference.nt")


def test_generate_default_config_dict_has_all_sections() -> None:
    cfg = generate_default_config_dict()
    assert "entity_alignment_configs" in cfg
    assert "metrics" in cfg
    assert "default" in cfg["entity_alignment_configs"]
    assert "verified_entities_path" in cfg["entity_alignment_configs"]["default"]

    metrics = cfg["metrics"]
    assert "entity_align" in metrics
    assert "duplicates" in metrics
    assert "triple_alignment" in metrics
    assert "consistency_violations" in metrics


def test_load_metric_configs_interpolates_vars_and_resolves_paths(tmp_path: Path) -> None:
    # mirror the style used in experiments/examples/scripts/run_eval.yaml
    cfg = tmp_path / "run_eval.yaml"
    (tmp_path / "test.ttl").write_text(
        """
@prefix : <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:a rdfs:label "A" .
""".lstrip(),
        encoding="utf-8",
    )
    cfg.write_text(
        """
reference_kg: test.ttl

entity_alignment_configs:
  default:
    method: label_embedding
    reference_kg: $reference_kg
    entity_sim_threshold: 0.95

metrics:
  duplicates:
    entity_alignment_config_ref: default
""".lstrip(),
        encoding="utf-8",
    )

    loaded = load_metric_configs(cfg)
    assert isinstance(loaded["duplicates"], DuplicateConfig)
    # reference_kg should be a KG whose path resolves relative to cfg location
    kg = loaded["duplicates"].entity_alignment_config.reference_kg
    assert kg is not None
    assert kg.path == (tmp_path / "test.ttl").resolve()

