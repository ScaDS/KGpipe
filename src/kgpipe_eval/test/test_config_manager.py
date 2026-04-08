from __future__ import annotations

from pathlib import Path

from kgpipe_eval.config.manager import load_metric_configs
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

