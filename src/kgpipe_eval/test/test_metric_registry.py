from kgpipe.common.discovery import (
    discover_entry_points,
    get_registered_metrics,
)
from kgpipe.common.registry import Registry
from kgpipe_eval.api import MeasurementSpec
from kgpipe_eval.metrics import METRIC_CLASSES, register_metrics
from kgpipe_eval.metrics.statistics import CountMetric


def test_register_metrics_adds_kgpipe_eval_catalog():
    register_metrics()

    registered = {getattr(m, "key", m.__name__) for m in get_registered_metrics()}
    expected = {cls.key for cls in METRIC_CLASSES}
    assert expected.issubset(registered)


def test_count_metric_declares_measurement_specs():
    assert CountMetric.key == "CountMetric"
    assert len(CountMetric.measurements) == 6
    by_name = {m.name: m.unit for m in CountMetric.measurements}
    assert by_name["entity_count"] == "number"
    assert by_name["class_occurrence"] == "dictionary"


def test_alignment_metrics_declare_distinct_aliases():
    from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric
    from kgpipe_eval.metrics.triple_alignment import TripleAlignmentMetric

    ea = {m.name: m for m in EntityAlignmentMetric.measurements}
    ta = {m.name: m for m in TripleAlignmentMetric.measurements}

    assert ea["precision"].alias == ("ACC_E",)
    assert ea["recall"].alias == ("COV_E",)
    assert ta["precision"].alias == ("ACC_T",)
    assert ta["recall"].alias == ("COV_T",)
    assert ea["precision"].alias != ta["precision"].alias
    assert ea["recall"].alias != ta["recall"].alias


def test_discover_entry_points_registers_eval_metrics():
    discover_entry_points()
    registered = Registry.list("metric")
    assert any(getattr(m, "key", None) == "CountMetric" for m in registered)
    assert any(getattr(m, "measurements", None) for m in registered)
