from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal
from collections import defaultdict
import re

def check_disjoint_class_violations(g: Graph):
    violations = []
    # Find all disjoint class pairs
    disjoints = set()
    for c1, _, c2 in g.triples((None, OWL.disjointWith, None)):
        disjoints.add((c1, c2))
        disjoints.add((c2, c1))  # Make symmetric

    # Map individuals to all their rdf:type classes
    class_map = defaultdict(set)
    for s, _, o in g.triples((None, RDF.type, None)):
        if o != OWL.Class:
            class_map[s].add(o)

    # Check for individuals typed with disjoint classes
    for ind, classes in class_map.items():
        for c1 in classes:
            for c2 in classes:
                if c1 != c2 and (c1, c2) in disjoints:
                    violations.append((ind, c1, c2))
    return violations

def check_domain_violations(g: Graph):
    violations = []
    for prop in set(g.subjects(RDF.type, RDF.Property)).union(g.subjects(RDF.type, OWL.ObjectProperty)):
        for s, p, o in g.triples((None, prop, None)):
            for _, _, domain in g.triples((prop, RDFS.domain, None)):
                if not (s, RDF.type, domain) in g:
                    violations.append((s, prop, "Expected domain", domain))
    return violations

def check_range_violations(g: Graph):
    violations = []
    for prop in set(g.subjects(RDF.type, RDF.Property)).union(g.subjects(RDF.type, OWL.ObjectProperty)):
        for s, p, o in g.triples((None, prop, None)):
            for _, _, range_ in g.triples((prop, RDFS.range, None)):
                if isinstance(o, Literal):
                    if (range_, RDF.type, RDFS.Datatype) not in g and not range_.startswith(str(XSD)):
                        violations.append((s, prop, "Expected non-literal", range_))
                elif not (o, RDF.type, range_) in g:
                    violations.append((o, prop, "Expected type", range_))
    return violations

def check_datatype_format_violations(g: Graph):
    violations = []
    for s, p, o in g.triples((None, None, None)):
        if isinstance(o, Literal):
            if o.datatype:
                try:
                    if o.datatype == XSD.integer:
                        int(o)
                    elif o.datatype == XSD.float:
                        float(o)
                    elif o.datatype == XSD.boolean:
                        assert o.value in [True, False]
                    elif o.datatype == XSD.date:
                        re.match(r"^\d{4}-\d{2}-\d{2}$", str(o))
                except Exception:
                    violations.append((s, p, o, "Invalid literal for", o.datatype))
    return violations

def check_cardinality_violations(g: Graph, cardinality_rules: dict):
    """
    cardinality_rules: {
      ex:hasChild: {
        "max": 2,
        "min": 1
      },
      ex:hasSpouse: {
        "exact": 1
      }
    }
    """
    violations = []
    for prop, rules in cardinality_rules.items():
        usage = defaultdict(list)
        for s, _, o in g.triples((None, prop, None)):
            usage[s].append(o)

        for s, values in usage.items():
            if "max" in rules and len(values) > rules["max"]:
                violations.append((s, prop, f"More than max ({rules['max']})", len(values)))
            if "min" in rules and len(values) < rules["min"]:
                violations.append((s, prop, f"Less than min ({rules['min']})", len(values)))
            if "exact" in rules and len(values) != rules["exact"]:
                violations.append((s, prop, f"Not exactly {rules['exact']}", len(values)))
    return violations

# from owlready2 import onto_path, Ontology, World, default_world, get_ontology, sync_reasoner



# onto = get_ontology("file:///home/marvin/project/code/experiments/src/geneval/test_data/inconsistent.rdf")

# onto.load()

# for ind in onto.individuals():
#     print(ind)
#     types = list(ind.is_a)

#     # Disjoint class violation check
#     for i in range(len(types)):
#         for j in range(i + 1, len(types)):
#             if types[j] in types[i].disjoints() or types[i] in types[j].disjoints():
#                 print(f"❌ Disjoint violation: {ind.name} is both {types[i].name} and {types[j].name}")
