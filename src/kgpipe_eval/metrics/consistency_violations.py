from kgpipe_eval.api import Metric, MetricResult, Measurement, MeasurementSpec

from pydantic import BaseModel, model_validator, ConfigDict
from kgpipe.common import KG
from pathlib import Path
from kgpipe_eval.utils.kg_utils import TripleGraph
from typing import ClassVar, Dict, Set, Optional

from rdflib import URIRef, RDF, Literal, Graph, XSD
from rdflib.query import Result, ResultRow

from kgcore.api.ontology import Ontology, OntologyUtil
from tqdm import tqdm

def get_ontology_graph(ontology_path: Optional[Path], kg: KG) -> Graph:
    if ontology_path is not None:
        return Graph().parse(ontology_path)
    elif kg is not None:
        return kg.get_ontology_graph()


def enrich_type_information(graph: Graph, ontology: Ontology, type_property: URIRef = RDF.type) -> Graph:
    type_dict = {}

    new_graph = Graph()

    for s, p, o in graph:
        domain, range = ontology.get_domain_range(str(p))
        if domain and isinstance(s, URIRef):
            if str(s) not in type_dict:
                type_dict[str(s)] = []
            type_dict[str(s)].append(str(domain))   
        if range and isinstance(o, URIRef):
            if str(o) not in type_dict:
                type_dict[str(o)] = []
            type_dict[str(o)].append(str(range))
        new_graph.add((s, p, o))

    for uri, types in type_dict.items():
        for type in types:
            new_graph.add((URIRef(uri), type_property, URIRef(type)))
    return new_graph

class ConsistencyViolationsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reference_kg: Optional[KG] = None
    ontology_path: Optional[Path] = None

    @model_validator(mode="after")
    def _require_reference_kg_or_ontology_path(self):
        if self.reference_kg is None and self.ontology_path is None:
            raise ValueError("Provide either `reference_kg` or `ontology_path`.")
        return self

class DisjointDomainMetric(Metric):
    key = "DisjointDomainMetric"
    description = "Subjects typed with mutually disjoint ontology classes."
    measurements: ClassVar[tuple[MeasurementSpec, ...]] = (
        MeasurementSpec(name="subjects_with_disjoint_domains", unit="number"),
        MeasurementSpec(name="subjects", unit="number"),
        MeasurementSpec(name="normalized_score", unit="ratio", alias=("O_DT")),
    )

    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        """Compute disjoint domain score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = get_ontology_graph(config.ontology_path, config.reference_kg)
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        for s, p, o in ontology_graph.triples((None, None, None)):
            graph.add((s, p, o))

        # Get all disjoint domains
        disjoint_domains_qr: Result = graph.query(
            """
            SELECT DISTINCT ?subject
            WHERE {
                ?subject a ?disjointDomain1 .
                ?subject a ?disjointDomain2 .
                ?disjointDomain1 owl:disjointWith ?disjointDomain2 .
            }
            """
        )
        subjects_with_disjoint_domains = set([row["subject"] for row in disjoint_domains_qr if isinstance(row, ResultRow)])

        subjects = set([str(s) for s in graph.subjects()])

        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="subjects_with_disjoint_domains", value=len(subjects_with_disjoint_domains), unit="number"),
                Measurement(name="subjects", value=len(subjects), unit="number"),
                Measurement(name="normalized_score", value=1.0 - (len(subjects_with_disjoint_domains) / len(subjects)), unit="ratio"),
            ],
            summary=f"Number of subjects with disjoint domains: {len(subjects_with_disjoint_domains)}",
        )

class DomainMetric(Metric):
    key = "DomainMetric"
    description = "Incorrect relation domain violations against an ontology."
    measurements: ClassVar[tuple[MeasurementSpec, ...]] = (
        MeasurementSpec(name="incorrect_relation_domain", unit="number"),
        MeasurementSpec(name="correct_relation_domain", unit="number"),
        MeasurementSpec(name="normalized_score", unit="ratio", alias=("O_D",)),
    )

    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        """Compute incorrect relation domain score.
        
        TODO: check if this is correct for increment eval if namespace changes to former generic namespace not seed
        """

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = get_ontology_graph(config.ontology_path, config.reference_kg)
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        # disjoint class by class
        disjoint_class_by_class : Dict[str, Set[str]] = {}
        for class_ in ontology.classes:
            if class_.disjointWith is not None:
                disjoint_class_by_class[class_.uri] = class_.disjointWith
            else:
                disjoint_class_by_class[class_.uri] = set()


        def is_subject_type(o, type):
            # print(o, type)
            if isinstance(o, URIRef):
                types = [str(t) for _, _, t in graph.triples((o, RDF.type, None))]
                return type in types and not any(str(other_type) in disjoint_class_by_class.get(str(type), set()) for other_type in types)
            elif isinstance(o, Literal):
                return o.datatype == type
            else:
                return False

        domain_by_property = {}
        for property in ontology.properties:
            if property.domain is not None:
                domain_by_property[property.uri] = property.domain.uri
            else:
                print(f"Property {property.uri} has no domain")
                domain_by_property[property.uri] = "TODO"

        incorrect_relation_domain = 0
        correct_relation_domain = 0

        for s, p, o in graph.triples((None, None, None)):
            if str(p) in domain_by_property:
                if is_subject_type(s, domain_by_property[str(p)]):
                    correct_relation_domain += 1
                else:
                    incorrect_relation_domain += 1

        if incorrect_relation_domain + correct_relation_domain > 0:
            normalized_score = 1.0 - (incorrect_relation_domain / (incorrect_relation_domain + correct_relation_domain))
        else:
            normalized_score = 0.0

        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="incorrect_relation_domain", value=incorrect_relation_domain, unit="number"),
                Measurement(name="correct_relation_domain", value=correct_relation_domain, unit="number"),
                Measurement(name="normalized_score", value=normalized_score, unit="ratio"),
            ],
            summary=f"Number of incorrect relation domain: {incorrect_relation_domain}",
            # name=self.name,
            # value=incorrect_relation_domain,
            # normalized_score=normalized_score,
            # details={"incorrect_relation_domain": incorrect_relation_domain, "correct_relation_domain": correct_relation_domain},
            # aspect=self.aspect
        )

class RangeMetric(Metric):
    key = "RangeMetric"
    description = "Incorrect relation range violations against an ontology."
    measurements: ClassVar[tuple[MeasurementSpec, ...]] = (
        MeasurementSpec(name="incorrect_relation_range", unit="number"),
        MeasurementSpec(name="correct_relation_range", unit="number"),
        MeasurementSpec(name="normalized_score", unit="ratio", alias=("O_R",)),
    )

    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        """Compute incorrect relation range score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = get_ontology_graph(config.ontology_path, config.reference_kg)
        ontology : Ontology= OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        # disjoint class by class
        disjoint_class_by_class : Dict[str, Set[str]] = {}
        for class_ in ontology.classes:
            if class_.disjointWith is not None:
                disjoint_class_by_class[class_.uri] = class_.disjointWith
            else:
                disjoint_class_by_class[class_.uri] = set()

        def is_object_type(o, type):
            # print(o, type)
            if isinstance(o, URIRef):
                types = [str(t) for s, p, t in graph.triples((o, RDF.type, None))]
                # if str(type) not in types:
                #     print(f"Incorrect relation range {types} of {o} for property {p} with range {types}")
                return str(type) in types and not any(str(other_type) in disjoint_class_by_class.get(str(type), set()) for other_type in types)
            elif isinstance(o, Literal):
                datatype = o.datatype
                if not datatype:
                    datatype = str(XSD.string)
                return str(datatype) == str(type)
            else:
                return False


        range_by_property = {}
        for property in ontology.properties:
            if property.range is not None:
                range_by_property[property.uri] = property.range.uri
            else:
                # print(f"Property {property.uri} has no range")
                range_by_property[property.uri] = None

        incorrect_relation_range = 0
        correct_relation_range = 0

        for s, p, o in graph.triples((None, None, None)):
            if str(p) in range_by_property:
                if is_object_type(o, range_by_property[str(p)]):
                    correct_relation_range += 1
                else:
                    # print(f"Incorrect relation range {o if isinstance(o, URIRef) else o.datatype} for property {p} with range {range_by_property[str(p)]}")
                    incorrect_relation_range += 1

        normalized_score = 1.0 - (incorrect_relation_range / (incorrect_relation_range + correct_relation_range)) if incorrect_relation_range + correct_relation_range > 0 else 1.0
        """Compute incorrect relation range score."""
        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="incorrect_relation_range", value=incorrect_relation_range, unit="number"),
                Measurement(name="correct_relation_range", value=correct_relation_range, unit="number"),
                Measurement(name="normalized_score", value=normalized_score, unit="ratio"),
            ],
            summary=f"Number of incorrect relation range: {incorrect_relation_range}",
            # name=self.name,
            # value=incorrect_relation_range,
            # normalized_score=normalized_score,
            # details={"incorrect_relation_range": incorrect_relation_range, "correct_relation_range": correct_relation_range},
            # aspect=self.aspect
        )

class RelationDirectionMetric(Metric):
    key = "RelationDirectionMetric"
    description = "Incorrect relation direction violations against an ontology."
    measurements: ClassVar[tuple[MeasurementSpec, ...]] = (
        MeasurementSpec(name="incorrect_relation_direction", unit="number"),
        MeasurementSpec(name="correct_relation_direction", unit="number"),
        MeasurementSpec(name="normalized_score", unit="ratio", alias=("O_RD",)),
    )

    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        """Compute incorrect relation direction score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = get_ontology_graph(config.ontology_path, config.reference_kg)
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        if len(ontology_graph) == 0:
            ontology_graph = graph
            print(f"INFO: ontology_graph is empty, using graph instead")

        # TODO use ontology implementation from framework
        predicate_defs_sr = ontology_graph.query(
            """
            SELECT DISTINCT ?predicate ?domain ?range
            WHERE {
                ?predicate rdfs:domain ?domain .
                ?predicate rdfs:range ?range .
            }
            """
        )

        # def check_type(uri, type):
        #     result = graph.query(
        #         """
        #         SELECT ?uri
        #         WHERE {
        #             ?uri a ?type .
        #         }
        #         """,
        #         initBindings={"uri": uri, "type": type}
        #     )
        #     return len(result) > 0

        predicate_defs = {}
        for row in predicate_defs_sr:
            predicate_defs[str(row["predicate"])] = (str(row["domain"]), str(row["range"]))

        incorrect_relation_direction = 0
        correct_relation_direction = 0

        entity_types = {}
        for s, p, o in graph.triples((None, RDF.type, None)):
            if str(s) not in entity_types:
                entity_types[str(s)] = []
            entity_types[str(s)].append(str(o))

        for s, p, o in tqdm(graph, desc="Checking relation direction"):
            if str(s) not in entity_types:
                continue
            if str(p) in predicate_defs:
                domain, range = predicate_defs[str(p)]

                if isinstance(o, URIRef):
                    if not str(s) in entity_types:
                        # print(f"Skipping s {s} because it is not in entity_types")
                        continue
                    if not str(o) in entity_types:
                        # print(f"Skipping o {o} because it is not in entity_types")
                        continue
                    if domain in entity_types[str(s)] and range in entity_types[str(o)]:
                        correct_relation_direction += 1
                    if domain in entity_types[str(o)] and range in entity_types[str(s)]:
                        incorrect_relation_direction += 1

        # print("incorrect_relation_direction", incorrect_relation_direction)
        # print("correct_relation_direction", correct_relation_direction)

        if incorrect_relation_direction + correct_relation_direction > 0:
            normalized_score = incorrect_relation_direction / (incorrect_relation_direction + correct_relation_direction)
            normalized_score = 1.0 - normalized_score
        else:
            normalized_score = 0.0

        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="incorrect_relation_direction", value=incorrect_relation_direction, unit="number"),
                Measurement(name="correct_relation_direction", value=correct_relation_direction, unit="number"),
                Measurement(name="normalized_score", value=normalized_score, unit="ratio"),
            ],
            summary=f"Number of incorrect relation direction: {incorrect_relation_direction}",
            # name=self.name,
            # value=incorrect_relation_direction,
            # normalized_score=normalized_score,
            # details={
            #     "incorrect_relation_direction": incorrect_relation_direction, 
            #     "correct_relation_direction": correct_relation_direction,
            #     "possible_relations": predicate_defs,
            #     "size_ontology_graph": len(ontology_graph)
            #     },
            # aspect=self.aspect
        )

class DatatypeMetric(Metric):
    key = "DatatypeMetric"
    description = "Incorrect datatype violations against an ontology."
    measurements: ClassVar[tuple[MeasurementSpec, ...]] = (
        MeasurementSpec(name="incorrect_datatype", unit="number"),
        MeasurementSpec(name="correct_datatype", unit="number"),
        MeasurementSpec(name="normalized_score", unit="ratio", alias=("O_LT",)),
    )

    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        """Compute incorrect datatype score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = get_ontology_graph(config.ontology_path, config.reference_kg)
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        def is_object_type(o, type):
            # print(o, type)
            if isinstance(o, URIRef):
                types = [str(t) for s, p, t in graph.triples((o, RDF.type, None))]
                # if str(type) not in types:
                #     print(f"Incorrect relation range {types} of {o} for property {p} with range {types}")
                return str(type) in types
            elif isinstance(o, Literal):
                datatype = o.datatype
                if not datatype:
                    datatype = str(XSD.string)
                return str(datatype) == str(type)
            else:
                return False
                
        # def is_object_type(o, type):
        #     # print(o, type)
        #     if isinstance(o, URIRef):
        #         types = [str(t) for s, p, t in graph.triples((o, RDF.type, None))]
        #         return type in types
        #     elif isinstance(o, Literal):
        #         return str(o.datatype) == type
        #     else:
        #         return False

        range_by_property = {}
        for property in ontology.properties:
            if property.range is not None:
                range_by_property[property.uri] = property.range.uri
            else:
                print(f"Property {property.uri} has no range")
                range_by_property[property.uri] = "TODO"

        incorrect_datatype = 0
        correct_datatype = 0

        for s, p, o in graph.triples((None, None, None)):
            if str(p) in range_by_property:
                if isinstance(o, Literal):
                    if not str(p) in range_by_property or is_object_type(o, range_by_property[str(p)]):
                        correct_datatype += 1
                    else:
                        incorrect_datatype += 1
                        # print(f"Incorrect datatype {o.datatype} for property {p} with range {range_by_property[str(p)]}")

        normalized_score = 1.0 - (incorrect_datatype / (incorrect_datatype + correct_datatype)) if incorrect_datatype + correct_datatype > 0 else 0.0

        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="incorrect_datatype", value=incorrect_datatype, unit="number"),
                Measurement(name="correct_datatype", value=correct_datatype, unit="number"),
                Measurement(name="normalized_score", value=normalized_score, unit="ratio"),
            ],
            summary=f"Number of incorrect datatype: {incorrect_datatype}",
            # name=self.name,
            # value=incorrect_datatype,
            # normalized_score=1.0 - (incorrect_datatype / (incorrect_datatype + correct_datatype)) if incorrect_datatype + correct_datatype > 0 else 0.0,
            # details={"incorrect_datatype": incorrect_datatype, "correct_datatype": correct_datatype},
            # aspect=self.aspect
        )

class DatatypeFormatMetric(Metric):
    key = "DatatypeFormatMetric"
    description = "Incorrect datatype format violations against an ontology."
    measurements: ClassVar[tuple[MeasurementSpec, ...]] = (
        MeasurementSpec(name="incorrect_datatype", unit="number"),
        MeasurementSpec(name="correct_datatype", unit="number"),
        MeasurementSpec(name="normalized_score", unit="ratio", alias=("O_LF",)),
    )

    def compute(self, kg: TripleGraph, config: ConsistencyViolationsConfig):
        """Compute incorrect datatype format score."""

        from kgpipe.evaluation.aspects.func.datatype_validator import validate_datatype

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = get_ontology_graph(config.ontology_path, config.reference_kg)
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        def is_object_type(o, type):
            # print(o, type)
            if isinstance(o, URIRef):
                types = [str(t) for s, p, t in graph.triples((o, RDF.type, None))]
                return type in types
            elif isinstance(o, Literal):
                return str(o.datatype) == type
            else:
                return False

        range_by_property = {}
        for property in ontology.properties:
            if property.range is not None:
                range_by_property[property.uri] = property.range.uri
            else:
                print(f"Property {property.uri} has no range")
                range_by_property[property.uri] = "TODO"

        incorrect_datatype = 0
        correct_datatype = 0

        for s, p, o in graph.triples((None, None, None)):
            if str(p) in range_by_property:
                if isinstance(o, Literal):
                    if str(p) in range_by_property:
                        if validate_datatype(str(o), range_by_property[str(p)]):
                            # print(f"Correct datatype {o.datatype} for property {p} and value {o} with range {range_by_property[str(p)]}")
                            correct_datatype += 1
                        else:
                            # print(f"Incorrect datatype {p} \'{o}\' {range_by_property[str(p)]}")
                            incorrect_datatype += 1
                    else:
                        print(f"Property {p} has no range")
                    # if not str(p) in range_by_property:
                    #     print(f"Property {p} has no range")
                    #     # or validate_datatype(str(o), range_by_property[str(p)]):
                    #     # print(f"Correct datatype {o.datatype} for property {p} and value {o} with range {range_by_property[str(p)]}")
                    #     correct_datatype += 1
                    # else:
                    #     incorrect_datatype += 1

        if incorrect_datatype + correct_datatype > 0:
            normalized_score = 1.0 - (incorrect_datatype / (incorrect_datatype + correct_datatype))
        else:
            normalized_score = 0.0

        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="incorrect_datatype", value=incorrect_datatype, unit="number"),
                Measurement(name="correct_datatype", value=correct_datatype, unit="number"),
                Measurement(name="normalized_score", value=normalized_score, unit="ratio"),
            ],
            summary=f"Number of incorrect datatype: {incorrect_datatype}",
            # name=self.name,
            # value=incorrect_datatype,
            # normalized_score=normalized_score,
            # details={"incorrect_datatype": incorrect_datatype, "correct_datatype": correct_datatype},
            # aspect=self.aspect
        )


# @Registry.metric()
# class OntologyClassCoverageMetric(Metric):
#     """Check if the KG has correct class coverage."""
#     def __init__(self):
#         super().__init__(
#             name="ontology_class_coverage",
#             description="Check if the KG has correct class coverage",
#             aspect=EvaluationAspect.SEMANTIC
#         )

#     def compute(self, kg: KG, config: SemanticConfig, **kwargs) -> MetricResult:
#         """Compute ontology class coverage score."""

#         raw_graph: Graph = kg.get_graph()
#         ontology_graph: Graph = kg.get_ontology_graph()
#         ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
#         graph = enrich_type_information(raw_graph, ontology)

#         expected_classes = set([c.uri for c in ontology.classes if not c.uri.startswith(str(OWL))])

#         found_classes = set(str(o) for s, p, o in graph.triples((None, RDF.type, None)) if not str(o).startswith(str(OWL)))

#         true_positive = len(expected_classes & found_classes)
#         false_positive = len(found_classes - expected_classes)
#         false_negative = len(expected_classes - found_classes)

#         precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
#         recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
#         f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

#         return MetricResult(
#             name=self.name,
#             value=true_positive,
#             normalized_score=f1_score,
#             details={"true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative},
#             aspect=self.aspect
#         )

# @Registry.metric()
# class OntologyRelationCoverageMetric(Metric):
#     """Check if the KG has correct relation coverage."""
#     def __init__(self):
#         super().__init__(
#             name="ontology_relation_coverage",
#             description="Check if the KG has correct relation coverage",
#             aspect=EvaluationAspect.SEMANTIC
#         )

#     def compute(self, kg: KG, config: SemanticConfig, **kwargs) -> MetricResult:
#         """Compute ontology relation coverage score."""

#         raw_graph: Graph = kg.get_graph()
#         ontology_graph: Graph = kg.get_ontology_graph()
#         ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
#         graph = enrich_type_information(raw_graph, ontology)

#         NOT_FILTER: List[str] = [str(OWL), str(RDF), str(RDFS)]

#         expected_relations = set([r.uri for r in ontology.properties])
#         expected_relations = set([r for r in expected_relations if not any(filter(lambda x: r.startswith(x), NOT_FILTER))])

#         # print(expected_relations)

#         found_relations = set(str(p) for _, p, _ in graph.triples((None, None, None)))
#         def filter_relation(r):
#             return any(filter(lambda x: r.startswith(x), NOT_FILTER))
#         found_relations = set([r for r in found_relations if not filter_relation(r)])

#         # print(found_relations)

#         true_positive = len(expected_relations & found_relations)
#         false_positive = len(found_relations - expected_relations)
#         false_negative = len(expected_relations - found_relations)

#         precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
#         recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
#         f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

#         return MetricResult(
#             name=self.name,
#             value=true_positive,
#             normalized_score=f1_score,
#             details={"true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative, "missing": (expected_relations - found_relations)},
#             aspect=self.aspect
#         )

# @Registry.metric()
# class OntologyPropertyCoverageMetric(Metric):
#     """Check if the KG has correct property coverage."""
#     def __init__(self):
#         super().__init__(
#             name="ontology_property_coverage",
#             description="Check if the KG has correct property coverage",
#             aspect=EvaluationAspect.SEMANTIC
#         )

#     def compute(self, kg: KG, config: SemanticConfig, **kwargs) -> MetricResult:
#         """Compute ontology property coverage score."""
#         return MetricResult(
#             name=self.name,
#             value=0.0,
#             normalized_score=1.0,
#             details={"error": "Not implemented"},
#             aspect=self.aspect
#         )

# @Registry.metric()
# class OntologyNamespaceCoverageMetric(Metric):
#     """Check if the KG has correct namespace coverage."""
#     def __init__(self):
#         super().__init__(
#             name="ontology_namespace_coverage",
#             description="Check if the KG has correct namespace coverage",
#             aspect=EvaluationAspect.SEMANTIC
#         )

#     def compute(self, kg: KG, config: SemanticConfig, **kwargs) -> MetricResult:
#         """Compute ontology namespace coverage score."""

#         # graph = kg.get_graph()
#         # ontology_graph = kg.get_ontology_graph()
#         # if len(ontology_graph) == 0:
#         #     ontology_graph = graph

#         # ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        

#         return MetricResult(
#             name=self.name,
#             value=0.0,
#             normalized_score=1.0,
#             details={"error": "Not implemented"},
#             aspect=self.aspect
#         )

# class OntologyClassCoverageMetric():
#     pass

# class OntologyRelationCoverageMetric():
#     pass

# class OntologyNamespaceCoverageMetric():
#     pass

# Cardinality Metric
        # """Compute incorrect relation cardinality score."""

        # raw_graph: Graph = kg.get_graph()
        # ontology_graph: Graph = kg.get_ontology_graph()
        # ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        # graph = enrich_type_information(raw_graph, ontology)
        # if len(ontology_graph) == 0:
        #     ontology_graph = graph

        # cardinality_by_property = {}
        # property_cardinalities: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # properties_in_graph = set()

        # for s, p, o in graph.triples((None, None, None)):
        #     properties_in_graph.add(str(p))

        # for property in properties_in_graph:
        #     cardinality_by_property[property] = get_property_cardinality(ontology_graph, property)

        # # print(cardinality_by_property)
        # # print(property_cardinalities)

        # for s, p, o in graph.triples((None, None, None)):
        #     if str(p) in cardinality_by_property:
        #         if str(s) in property_cardinalities[str(p)]:
        #             property_cardinalities[str(p)][str(s)] += 1
        #         else:
        #             property_cardinalities[str(p)][str(s)] = 1

        # incorrect_cardinality = 0
        # correct_cardinality = 0

        # for property, cardinality in property_cardinalities.items():
        #     min, max = cardinality_by_property[property]
        #     for subject, count in cardinality.items():
        #         if count > max:
        #             incorrect_cardinality += 1
        #         elif count < min:
        #             incorrect_cardinality += 1
        #         else:
        #             correct_cardinality += 1

        # return MetricResult(
        #     name=self.name,
        #     value=incorrect_cardinality,
        #     normalized_score=1.0 - (incorrect_cardinality / (incorrect_cardinality + correct_cardinality)) if incorrect_cardinality + correct_cardinality > 0 else 0.0,
        #     details={"incorrect_cardinality": incorrect_cardinality, "correct_cardinality": correct_cardinality},
        #     aspect=self.aspect
        # )