"""
Semantic Evaluation Aspect

Evaluates semantic properties and consistency of knowledge graphs.
"""

from typing import Dict, List, Optional, Any
import json
from rdflib import Graph, URIRef, RDF, Literal, OWL, RDFS, XSD
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib.query import Result, ResultRow
from pathlib import Path
from kgpipe.execution.docker import docker_client
from kgpipe.common.io import get_docker_volume_bindings
from ...common.models import KG, Data, DataFormat
from ..base import EvaluationAspect, AspectResult, AspectEvaluator, Metric, MetricResult
from kgpipe.common.io import remap_data_path_for_container
from tqdm import tqdm
from collections import defaultdict
from kgcore.model.ontology import OntologyExtractor, OntologyUtil, Ontology
from kgpipe.common.registry import Registry
import time


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

@Registry.metric()
class ReasoningMetric(Metric):
    """Check if the KG is reasoning."""
    
    def __init__(self):
        super().__init__(
            name="reasoning",
            description="Check if the KG is reasoning",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute reasoning score."""

        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nt")
        tmpfile = tmp.name

        combined_graph = Graph()
        for triple in kg.get_graph():
            combined_graph.add(triple)
        for triple in kg.get_ontology_graph():
            combined_graph.add(triple)

        combined_graph.serialize(destination=tmpfile, format="nt")

        all_data = list([Data(tmpfile, DataFormat.RDF_NTRIPLES)])
        volumes, host_to_container = get_docker_volume_bindings(all_data)

        mapped_data = remap_data_path_for_container(all_data[0], host_to_container)

        # Run the Pellet consistency check
        client = docker_client(
            image="kgt/pellet:latest",
            command=["-c", "cli/target/pelletcli/bin/pellet consistency -l Jena " + str(mapped_data.path) + " 2>&1 | tail -n 1"], # ["cli/target/pelletcli/bin/pellet", "consistency", "-l", "Jena", tmpfile],
            entrypoint="/bin/bash",
            volumes=volumes,
        )

        tmp.close()

        stdout = client()

        # print(stdout)

        # Parse the output
        value = 0.0
        if stdout == "Consistent: Yes\n":
            value = 1.0
        else:
            value = 0.0

        # Clean up the temporary file

        return MetricResult(
            name=self.name,
            value=value,
            normalized_score=value,
            details={"stdout": stdout},
            aspect=self.aspect
        )

@Registry.metric()
class SchemaConsistencyMetric(Metric):
    """Check basic schema consistency of the KG."""
    pass

    def __init__(self):
        super().__init__(
            name="schema_consistency",
            description="Basic schema consistency check",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute schema consistency score."""
        try:
            # Simple implementation - check for basic RDF structure
            consistency_score = 0.0
            details = {"checks": []}
            
            if kg.format.value in ['ttl', 'rdf', 'jsonld']:
                with open(kg.path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    # Check for valid triple structure
                    valid_triples = 0
                    total_lines = 0
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            total_lines += 1
                            parts = line.split()
                            if len(parts) >= 3:
                                valid_triples += 1
                    
                    if total_lines > 0:
                        consistency_score = valid_triples / total_lines
                        details["checks"].append({
                            "check": "triple_structure",
                            "valid_triples": valid_triples,
                            "total_lines": total_lines,
                            "score": consistency_score
                        })
            
            elif kg.format.value == 'json':
                with open(kg.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Check for required fields in JSON structure
                    required_fields = ['subject', 'predicate', 'object']
                    if isinstance(data, list) and data:
                        sample_item = data[0]
                        if isinstance(sample_item, dict):
                            present_fields = sum(1 for field in required_fields if field in sample_item)
                            consistency_score = present_fields / len(required_fields)
                            details["checks"].append({
                                "check": "json_structure",
                                "present_fields": present_fields,
                                "required_fields": len(required_fields),
                                "score": consistency_score
                            })
            
            return MetricResult(
                name=self.name,
                value=consistency_score,
                normalized_score=consistency_score,  # Already 0-1
                details=details,
                aspect=self.aspect
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                normalized_score=0.0,
                details={"error": str(e)},
                aspect=self.aspect
            )

@Registry.metric()
class NamespaceUsageMetric(Metric):
    """Check for proper namespace usage in the KG."""

    def __init__(self):
        super().__init__(
            name="namespace_usage",
            description="Check for proper namespace usage",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute namespace usage score."""
        try:
            namespaces = set()
            total_entities = 0
            
            if kg.format.value in ['ttl', 'rdf', 'jsonld']:
                with open(kg.path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 3:
                                # Extract namespace from URIs
                                for part in [parts[0], parts[1], parts[2]]:
                                    if part.startswith('<') and part.endswith('>'):
                                        uri = part[1:-1]
                                        if '#' in uri:
                                            namespace = uri.split('#')[0]
                                            namespaces.add(namespace)
                                        elif '/' in uri:
                                            namespace = '/'.join(uri.split('/')[:-1])
                                            namespaces.add(namespace)
                                        total_entities += 1
            
            # Score based on namespace diversity (more namespaces = better)
            if total_entities > 0:
                # Normalize: 1+ namespaces is good, 0 is bad
                score = self.normalize_score(len(namespaces), 0, 5)
            else:
                score = 0.0
            
            return MetricResult(
                name=self.name,
                value=float(len(namespaces)),
                normalized_score=score,
                details={
                    "namespaces": list(namespaces),
                    "namespace_count": len(namespaces),
                    "total_entities": total_entities
                },
                aspect=self.aspect
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                normalized_score=0.0,
                details={"error": str(e)},
                aspect=self.aspect
            )

@Registry.metric()
class DisjointDomainMetric(Metric):
    """Check if the KG has disjoint domains."""
    
    def __init__(self):
        super().__init__(
            name="disjoint_domain",
            description="Check if the KG has disjoint domains",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute disjoint domain score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
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
            name=self.name,
            value=len(subjects_with_disjoint_domains),
            normalized_score=1.0 - (len(subjects_with_disjoint_domains) / len(subjects)),
            details={"subjects_with_disjoint_domains": len(subjects_with_disjoint_domains), "subjects": len(subjects)},
            aspect=self.aspect
        )

@Registry.metric()
class IncorrectRelationDirectionMetric(Metric):
    """Number of incorrect relation direction."""
    
    def __init__(self):
        super().__init__(
            name="incorrect_relation_direction",
            description="Check if the KG has incorrect relation direction",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute incorrect relation direction score."""
        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        if len(ontology_graph) == 0:
            ontology_graph = graph

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
            name=self.name,
            value=incorrect_relation_direction,
            normalized_score=normalized_score,
            details={
                "incorrect_relation_direction": incorrect_relation_direction, 
                "correct_relation_direction": correct_relation_direction,
                "possible_relations": predicate_defs,
                "size_ontology_graph": len(ontology_graph)
                },
            aspect=self.aspect
        )

from kgpipe.evaluation.aspects.func.ontology_func import get_property_cardinality

@Registry.metric()
class IncorrectRelationCardinalityMetric(Metric):
    """Number of incorrect relation cardinality."""
    def __init__(self):
        super().__init__(
            name="incorrect_relation_cardinality",
            description="Check if the KG has incorrect relation cardinality",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute incorrect relation cardinality score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)
        if len(ontology_graph) == 0:
            ontology_graph = graph

        cardinality_by_property = {}
        property_cardinalities: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        properties_in_graph = set()

        for s, p, o in graph.triples((None, None, None)):
            properties_in_graph.add(str(p))

        for property in properties_in_graph:
            cardinality_by_property[property] = get_property_cardinality(ontology_graph, property)

        # print(cardinality_by_property)
        # print(property_cardinalities)

        for s, p, o in graph.triples((None, None, None)):
            if str(p) in cardinality_by_property:
                if str(s) in property_cardinalities[str(p)]:
                    property_cardinalities[str(p)][str(s)] += 1
                else:
                    property_cardinalities[str(p)][str(s)] = 1

        incorrect_cardinality = 0
        correct_cardinality = 0

        for property, cardinality in property_cardinalities.items():
            min, max = cardinality_by_property[property]
            for subject, count in cardinality.items():
                if count > max:
                    incorrect_cardinality += 1
                elif count < min:
                    incorrect_cardinality += 1
                else:
                    correct_cardinality += 1

        return MetricResult(
            name=self.name,
            value=incorrect_cardinality,
            normalized_score=1.0 - (incorrect_cardinality / (incorrect_cardinality + correct_cardinality)) if incorrect_cardinality + correct_cardinality > 0 else 0.0,
            details={"incorrect_cardinality": incorrect_cardinality, "correct_cardinality": correct_cardinality},
            aspect=self.aspect
        )

@Registry.metric()
class IncorrectRelationRangeMetric(Metric):
    """Number of incorrect relation range."""
    def __init__(self):
        super().__init__(
            name="incorrect_relation_range",
            description="Check if the KG has incorrect relation range",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
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
            name=self.name,
            value=incorrect_relation_range,
            normalized_score=normalized_score,
            details={"incorrect_relation_range": incorrect_relation_range, "correct_relation_range": correct_relation_range},
            aspect=self.aspect
        )

@Registry.metric()
class IncorrectRelationDomainMetric(Metric):
    """Number of incorrect relation domain."""
    def __init__(self):
        super().__init__(
            name="incorrect_relation_domain",
            description="Check if the KG has incorrect relation domain",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute incorrect relation domain score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
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
            name=self.name,
            value=incorrect_relation_domain,
            normalized_score=normalized_score,
            details={"incorrect_relation_domain": incorrect_relation_domain, "correct_relation_domain": correct_relation_domain},
            aspect=self.aspect
        )

@Registry.metric()
class IncorrectDatatypeMetric(Metric):
    """Number of incorrect datatype."""
    def __init__(self):
        super().__init__(
            name="incorrect_datatype",
            description="Check if the KG has incorrect datatype",
            aspect=EvaluationAspect.SEMANTIC
        )
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute incorrect datatype score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
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

        return MetricResult(
            name=self.name,
            value=incorrect_datatype,
            normalized_score=1.0 - (incorrect_datatype / (incorrect_datatype + correct_datatype)) if incorrect_datatype + correct_datatype > 0 else 0.0,
            details={"incorrect_datatype": incorrect_datatype, "correct_datatype": correct_datatype},
            aspect=self.aspect
        )

@Registry.metric()
class IncorrectDatatypeFormatMetric(Metric):
    """Number of incorrect datatype format."""
    def __init__(self):
        super().__init__(
            name="incorrect_datatype_format",
            description="Check if the KG has incorrect datatype format",
            aspect=EvaluationAspect.SEMANTIC
        )

    
    
    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute incorrect datatype format score."""

        from kgpipe.evaluation.aspects.func.datatype_validator import validate_datatype

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
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
            name=self.name,
            value=incorrect_datatype,
            normalized_score=normalized_score,
            details={"incorrect_datatype": incorrect_datatype, "correct_datatype": correct_datatype},
            aspect=self.aspect
        )

# check format if datatype is not set but infer from ontology
@Registry.metric()
class OntologyClassCoverageMetric(Metric):
    """Check if the KG has correct class coverage."""
    def __init__(self):
        super().__init__(
            name="ontology_class_coverage",
            description="Check if the KG has correct class coverage",
            aspect=EvaluationAspect.SEMANTIC
        )

    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute ontology class coverage score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        expected_classes = set([c.uri for c in ontology.classes if not c.uri.startswith(str(OWL))])

        found_classes = set(str(o) for s, p, o in graph.triples((None, RDF.type, None)) if not str(o).startswith(str(OWL)))

        true_positive = len(expected_classes & found_classes)
        false_positive = len(found_classes - expected_classes)
        false_negative = len(expected_classes - found_classes)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=true_positive,
            normalized_score=f1_score,
            details={"true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative},
            aspect=self.aspect
        )

@Registry.metric()
class OntologyRelationCoverageMetric(Metric):
    """Check if the KG has correct relation coverage."""
    def __init__(self):
        super().__init__(
            name="ontology_relation_coverage",
            description="Check if the KG has correct relation coverage",
            aspect=EvaluationAspect.SEMANTIC
        )

    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute ontology relation coverage score."""

        raw_graph: Graph = kg.get_graph()
        ontology_graph: Graph = kg.get_ontology_graph()
        ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        graph = enrich_type_information(raw_graph, ontology)

        NOT_FILTER: List[str] = [str(OWL), str(RDF), str(RDFS)]

        expected_relations = set([r.uri for r in ontology.properties])
        expected_relations = set([r for r in expected_relations if not any(filter(lambda x: r.startswith(x), NOT_FILTER))])

        # print(expected_relations)

        found_relations = set(str(p) for _, p, _ in graph.triples((None, None, None)))
        def filter_relation(r):
            return any(filter(lambda x: r.startswith(x), NOT_FILTER))
        found_relations = set([r for r in found_relations if not filter_relation(r)])

        # print(found_relations)

        true_positive = len(expected_relations & found_relations)
        false_positive = len(found_relations - expected_relations)
        false_negative = len(expected_relations - found_relations)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=true_positive,
            normalized_score=f1_score,
            details={"true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative, "missing": (expected_relations - found_relations)},
            aspect=self.aspect
        )

@Registry.metric()
class OntologyPropertyCoverageMetric(Metric):
    """Check if the KG has correct property coverage."""
    def __init__(self):
        super().__init__(
            name="ontology_property_coverage",
            description="Check if the KG has correct property coverage",
            aspect=EvaluationAspect.SEMANTIC
        )

    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute ontology property coverage score."""
        return MetricResult(
            name=self.name,
            value=0.0,
            normalized_score=1.0,
            details={"error": "Not implemented"},
            aspect=self.aspect
        )

@Registry.metric()
class OntologyNamespaceCoverageMetric(Metric):
    """Check if the KG has correct namespace coverage."""
    def __init__(self):
        super().__init__(
            name="ontology_namespace_coverage",
            description="Check if the KG has correct namespace coverage",
            aspect=EvaluationAspect.SEMANTIC
        )

    def compute(self, kg: KG, **kwargs) -> MetricResult:
        """Compute ontology namespace coverage score."""

        # graph = kg.get_graph()
        # ontology_graph = kg.get_ontology_graph()
        # if len(ontology_graph) == 0:
        #     ontology_graph = graph

        # ontology = OntologyUtil.load_ontology_from_graph(ontology_graph)
        

        return MetricResult(
            name=self.name,
            value=0.0,
            normalized_score=1.0,
            details={"error": "Not implemented"},
            aspect=self.aspect
        )

class SemanticEvaluator(AspectEvaluator):
    """Evaluator for semantic aspects of knowledge graphs."""
    
    def __init__(self):
        super().__init__(EvaluationAspect.SEMANTIC)
        self.metrics = [
            # ReasoningMetric(),
            DisjointDomainMetric(),
            IncorrectRelationDirectionMetric(),
            IncorrectRelationCardinalityMetric(),
            IncorrectRelationRangeMetric(),
            IncorrectRelationDomainMetric(),
            IncorrectDatatypeMetric(),
            IncorrectDatatypeFormatMetric(),
            OntologyClassCoverageMetric(),
            OntologyRelationCoverageMetric(),
            # OntologyPropertyCoverageMetric(),
            OntologyNamespaceCoverageMetric(),
        ]
    
    def evaluate(self, kg: KG, metrics: Optional[List[str]] = None, **kwargs) -> AspectResult:
        """Evaluate semantic properties of the KG."""
        results = []
        
        # Filter metrics if specified
        metrics_to_compute = self.metrics
        if metrics:
            metrics_to_compute = [m for m in self.metrics if m.name in metrics]
        
        # Compute each metric
        for metric in metrics_to_compute:
            try:
                start_time = time.time()
                result = metric.compute(kg, **kwargs)
                end_time = time.time()
                result.duration = end_time - start_time
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = MetricResult(
                    name=metric.name,
                    value=0.0,
                    normalized_score=0.0,
                    details={"error": str(e)},
                    aspect=self.aspect
                )
                results.append(error_result)
        
        # Calculate overall score as average of normalized scores
        if results:
            overall_score = sum(r.normalized_score for r in results) / len(results)
        else:
            overall_score = 0.0
        
        return AspectResult(
            aspect=self.aspect,
            metrics=results,
            overall_score=overall_score,
            details={
                "total_metrics": len(results),
                "successful_metrics": len([r for r in results if "error" not in r.details])
            }
        )
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available semantic metrics."""
        return [metric.name for metric in self.metrics] 