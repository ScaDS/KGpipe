import pytest
from kgpipe.common import KG, DataFormat
from pathlib import Path

from kgpipe.evaluation.aspects.statistical import EntityCountMetric, RelationCountMetric, TripleCountMetric , ClassCountMetric, ClassOccurrenceMetric, RelationOccurrenceMetric, PropertyOccurrenceMetric, NamespaceUsageMetric
from kgpipe.evaluation.writer import render_as_table, render_metric_as_table
from kgpipe.test.util import get_test_data_path, get_test_kg

kg = get_test_kg("product_kg.ttl")

SHOW_DETAILS = True

#==============================================
# Count Metrics
#==============================================

def test_entity_count():
    metric = EntityCountMetric()
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

def test_relation_count():
    metric = RelationCountMetric()
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

def test_triple_count():
    metric = TripleCountMetric()
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

def test_entity_type_count():
    metric = ClassCountMetric()
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

#==============================================
# Occurrence Metrics
#==============================================

def test_class_occurrence():
    metric = ClassOccurrenceMetric()
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

def test_relation_occurrence():
    metric = RelationOccurrenceMetric()
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

# TODO: all, object properties, data properties

def test_property_occurrence():
    metric = PropertyOccurrenceMetric()
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

#==============================================
# Namespace Usage Metrics
#==============================================

def test_namespace_usage():
    metric = NamespaceUsageMetric()
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)
