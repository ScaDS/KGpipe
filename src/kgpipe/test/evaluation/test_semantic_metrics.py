from kgpipe.evaluation.aspects.semantic import (
    ReasoningMetric, 
    
    DisjointDomainMetric, 
    
    IncorrectDatatypeFormatMetric,
    IncorrectRelationCardinalityMetric,
    IncorrectRelationDirectionMetric,
    IncorrectRelationRangeMetric,
    IncorrectRelationDomainMetric,
    IncorrectDatatypeMetric,
    IncorrectRelationCardinalityMetric,
    IncorrectRelationDirectionMetric,

    OntologyClassCoverageMetric,
    OntologyRelationCoverageMetric,
    OntologyPropertyCoverageMetric,
    OntologyNamespaceCoverageMetric,
)

from kgpipe.evaluation.writer import render_metric_as_table
from kgpipe.test.util import get_test_kg
import pytest

SHOW_DETAILS = True

#==============================================
# Reasoning Metrics
#==============================================

def test_reasoning():
    metric = ReasoningMetric()
    report = metric.compute(get_test_kg("consistency/inconsistent.rdf"))
    render_metric_as_table(report, show_details=SHOW_DETAILS)

    assert report.value == 0.0

#==============================================
# Consistency Metrics
#==============================================

def test_disjoint_domain():
    metric = DisjointDomainMetric()
    kg = KG(id="test", name="test", path=Path("/home/marvin/project/data/out/large/text_a/stage_3/result.nt"), format=DataFormat.RDF_NTRIPLES)
    ontology_path = Path("/home/marvin/project/code/experiments/movie-ontology.ttl")
    ontology_graph = Graph()
    ontology_graph.parse(ontology_path, format="turtle")
    kg.set_ontology_graph(ontology_graph)
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

    # assert report.value == 1

def test_range():
    metric = IncorrectRelationRangeMetric()
    # report = metric.compute(get_test_kg("consistency/incorrect_range.ttl"))
    kg = KG(id="test", name="test", path=Path("/home/marvin/project/data/out/large/text_a/stage_3/result.nt"), format=DataFormat.RDF_NTRIPLES)
    ontology_path = Path("/home/marvin/project/code/experiments/movie-ontology.ttl")
    ontology_graph = Graph()
    ontology_graph.parse(ontology_path, format="turtle")
    kg.set_ontology_graph(ontology_graph)
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

def test_domain():
    metric = IncorrectRelationDomainMetric()
    report = metric.compute(get_test_kg("consistency/incorrect_domain.ttl"))
    render_metric_as_table(report, show_details=SHOW_DETAILS)

def test_datatype():
    metric = IncorrectDatatypeMetric()
    report = metric.compute(get_test_kg("consistency/incorrect_datatype.ttl"))
    render_metric_as_table(report, show_details=SHOW_DETAILS)

    assert report.value == 1

from kgpipe.common import KG, DataFormat
from pathlib import Path
from rdflib import Graph

# @pytest.mark.skip(reason="Not implemented yet")
def test_datatype_format():
    metric = IncorrectDatatypeFormatMetric()
    kg = KG(id="test", name="test", path=Path("/home/marvin/project/data/out/large/text_a/stage_3/result.nt"), format=DataFormat.RDF_NTRIPLES)
    ontology_path = Path("/home/marvin/project/code/experiments/movie-ontology.ttl")
    ontology_graph = Graph()
    ontology_graph.parse(ontology_path, format="turtle")
    kg.set_ontology_graph(ontology_graph)
    # report = metric.compute(get_test_kg("consistency/incorrect_datatype_format.ttl"))
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

def test_cardinality():
    metric = IncorrectRelationCardinalityMetric()
    kg = KG(id="test", name="test", path=Path("/home/marvin/project/data/out/large/text_a/stage_3/result.nt"), format=DataFormat.RDF_NTRIPLES)
    ontology_path = Path("/home/marvin/project/code/experiments/movie-ontology.ttl")
    ontology_graph = Graph()
    ontology_graph.parse(ontology_path, format="turtle")
    kg.set_ontology_graph(ontology_graph)
    report = metric.compute(kg)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

    # assert report.value == 1

def test_relation_direction():
    metric = IncorrectRelationDirectionMetric()
    report = metric.compute(get_test_kg("consistency/incorrect_relation_direction.ttl"))
    render_metric_as_table(report, show_details=SHOW_DETAILS)

#==============================================
# Coverage Metrics (Ontology)
#==============================================

def test_class_coverage():
    metric = OntologyClassCoverageMetric()
    report = metric.compute(get_test_kg("consistency/ontology_coverage.ttl"))
    render_metric_as_table(report, show_details=SHOW_DETAILS)

    assert f"{report.value:.2f}" == "0.50"

def test_relation_coverage():
    metric = OntologyRelationCoverageMetric()
    report = metric.compute(get_test_kg("consistency/ontology_coverage.ttl"))
    render_metric_as_table(report, show_details=SHOW_DETAILS)

    assert f"{report.value:.2f}" == "0.50"

@pytest.mark.skip(reason="Not implemented yet")
def test_property_coverage():
    metric = OntologyPropertyCoverageMetric()
    report = metric.compute(get_test_kg("consistency/ontology_coverage.ttl"))
    render_metric_as_table(report, show_details=SHOW_DETAILS)

@pytest.mark.skip(reason="Not implemented yet")
def test_namespace_coverage():
    metric = OntologyNamespaceCoverageMetric()
    report = metric.compute(get_test_kg("consistency/ontology_coverage.ttl"))
    render_metric_as_table(report, show_details=SHOW_DETAILS)

#==============================================
# Unit Tests on the validator_rdflib.py
#==============================================

# from kgpipe.evaluation.aspects.semanticlib.validator_rdflib import check_disjoint_class_violations, check_domain_violations, check_range_violations, check_datatype_format_violations, check_cardinality_violations
# from kgpipe.evaluation.aspects.semanticlib.util import extract_cardinality_restrictions

# from rdflib import Graph, RDF, OWL, Literal, BNode, Graph, URIRef, XSD
# import pytest

# from kgpipe.test.util import get_test_data_path

# @pytest.mark.skip(reason="Not implemented yet")
# def test_disjoint_class_violations():
#     g = Graph()
#     g.parse(get_test_data_path("inconsistent.rdf"), format="xml")
#     violations = check_disjoint_class_violations(g)

#     print(violations)
#     assert len(violations) == 2
#     # assert violations[0][0] == "http://example.org#Cat"
#     # assert violations[0][1] == "http://example.org#Dog"

# @pytest.mark.skip(reason="Not implemented yet")
# def test_domain_violations():
#     g = Graph()
#     g.parse(get_test_data_path("incorrect_domain.ttl"), format="turtle")
#     violations = check_domain_violations(g)

#     print(violations)
#     assert len(violations) == 1
#     # assert violations[0][0] == "http://example.org#Cat"
#     # assert violations[0][1] == "http://example.org#Dog"

# @pytest.mark.skip(reason="Not implemented yet")
# def test_range_violations():
#     g = Graph()
#     g.parse(get_test_data_path("incorrect_range.ttl"), format="turtle")
#     violations = check_range_violations(g)

#     print(violations)
#     assert len(violations) == 1
#     # assert violations[0][0] == "http://example.org#Cat"
#     # assert violations[0][1] == "http://example.org#Dog"

# @pytest.mark.skip(reason="Check Impl")
# def test_datatype_format_violations():
#     g = Graph()
#     g.parse(get_test_data_path("incorrect_datatype_format.ttl"), format="turtle")
#     violations = check_datatype_format_violations(g)

#     print(violations)
#     assert len(violations) == 1
#     # assert violations[0][0] == "http://example.org#Cat"
#     # assert violations[0][1] == "http://example.org#Dog"

# @pytest.mark.skip(reason="Not implemented yet")
# def test_cardinality_violations():
#     g = Graph()
#     g.parse(get_test_data_path("incorrect_cardinality.ttl"), format="turtle")
#     violations = check_cardinality_violations(g, extract_cardinality_restrictions(g))

#     print(violations)
#     assert len(violations) == 1
#     # assert violations[0][0] == "http://example.org#Cat"
#     # assert violations[0][1] == "http://example.org#Dog"
