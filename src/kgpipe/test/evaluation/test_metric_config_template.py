import yaml
from pathlib import Path

from kgpipe.evaluation.aspects.reference import ReferenceConfig
from kgpipe.evaluation.util import get_metric_config_template, read_metric_config_yaml


def test_get_metric_config_template_for_reference_config():
    template_yaml = get_metric_config_template(ReferenceConfig)
    template = yaml.safe_load(template_yaml)

    assert template["name"] is None
    assert template["GT_MATCHES"] is None
    assert template["GT_MATCHES_TARGET_DATASET"] is None
    assert template["ENTITY_MATCH_THRESHOLD"] == 0.5
    assert template["RELATION_MATCH_THRESHOLD"] == 0.5
    assert template["VERIFIED_SOURCE_ENTITIES"] is None
    assert template["REFERENCE_KG_PATH"] is None
    assert template["EXPECTED_TEXT_LINKS"] is None
    assert template["TE_LINK_THRESHOLD"] == 0.4
    assert template["SEED_KG_PATH"] is None
    assert template["source_meta"] is None
    assert template["dataset"] is None
    assert template["JSON_EXPECTED_DIR"] is None
    assert template["JSON_EXPECTED_RELATION_FILE"] is None


def test_metric_config_template_roundtrip_reference_config(tmp_path: Path):
    template_yaml = get_metric_config_template(ReferenceConfig)
    template = yaml.safe_load(template_yaml)
    template["name"] = "reference-config-roundtrip"
    template["GT_MATCHES"] = "/tmp/gt_matches.csv"

    config_path = tmp_path / "reference_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(template, f, sort_keys=False)

    config = read_metric_config_yaml(config_path.as_posix(), ReferenceConfig)

    assert isinstance(config, ReferenceConfig)
    assert config.name == "reference-config-roundtrip"
    assert config.GT_MATCHES == Path("/tmp/gt_matches.csv")
    assert config.ENTITY_MATCH_THRESHOLD == 0.5
    assert config.TE_LINK_THRESHOLD == 0.4
