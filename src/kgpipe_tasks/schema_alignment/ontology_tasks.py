# from pydantic import BaseModel
# from rdflib import Graph, URIRef, OWL, RDFS, RDF 
# from collections import defaultdict
# import json

# def extract_owl(rdf: Graph):
#     graph: Graph = rdf

#     filter_predicates = set(str(OWL),str(RDFS), str(RDF))

#     owl_graph = Graph()

#     for s, p, o in graph:
#         # is owl, rdf or rdfs predicate
#         p_uri = str(p)
#         if "#" in p_uri:
#             p_uri = p_uri.split("#")[0]
#         else:
#             p_uri = p_uri.split("/")[0]
#         if p_uri in filter_predicates:
#             owl_graph.add(s,p,o)

#     return owl_graph

# def generate_owl(rdf: Graph):
#     pass

# def ontology_infer_types(i,o):
#     pass

# def infereSubjectTypesFromProperties(rdf: Graph, owl_graph: Graph, top_k=None):
#     """will use the rdfs:domain specs to infer subject types from properties"""
#     domain_by_property = {}
    
#     # Extract domains for OWL properties
#     for prop, _, prop_type in owl_graph.triples((None, RDF.type, None)):
#         if prop_type in (OWL.DatatypeProperty, OWL.ObjectProperty):
#             for _, _, domain in owl_graph.triples((prop, RDFS.domain, None)):
#                 domain_by_property[str(prop)] = str(domain)

#     # Collect possible subject types from RDF data
#     subject_type_counts = defaultdict((lambda: defaultdict(int)))

#     for subj, pred, _ in rdf:
#         pred_str = str(pred)
#         subj_str = str(subj)
#         if pred_str in domain_by_property:
#             domain = domain_by_property[pred_str]
#             subject_type_counts[subj_str][domain] += 1

#     # Normalize counts to proportions
#     subject_type_proportions = {}
#     for subj, type_counts in subject_type_counts.items():
#         total = sum(type_counts.values())
#         sorted_types = sorted(
#             ((t, count / total) for t, count in type_counts.items()),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         if top_k:
#             sorted_types = sorted_types[:top_k]
#         subject_type_proportions[subj] = sorted_types
#     return subject_type_proportions


# if __name__ == "__main__":

#     EXAMPLE_RDF_TTL="""
#     @prefix ex: <http://example.org/> .
#     @prefix foaf: <http://xmlns.com/foaf/0.1/> .
#     ex:Alice ex:birthYear "1996" ;
#         ex:name "Alice" ;
#         ex:released "2022" .
#     ex:Titanic ex:released "1997" .
#     ex:Misc foaf:name "Misc" .
#     """

#     EXAMPLE_OWL_TTL="""
#     @prefix ex: <http://example.org/> .
#     @prefix owl: <http://www.w3.org/2002/07/owl#> .
#     @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
#     @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
#     ex:Person a owl:Class ;
#         rdfs:subClassOf ex:Agent .
#     ex:name a owl:DatatypeProperty ;
#         rdfs:domain ex:Person ;
#         rdfs:range rdfs:Literal .
#     ex:birthYear a owl:DatatypeProperty ;
#         rdfs:domain ex:Person ;
#         rdfs:range xsd:integer .
#     ex:Film a owl:Class ;
#         rdfs:subClassOf ex:Media .
#     ex:released a owl:DatatypeProperty ;
#         rdfs:domain ex:Film ;
#         rdfs:range xsd:integer .
#     """

#     rdf = Graph()
#     rdf.parse(data=EXAMPLE_RDF_TTL, format="turtle")

#     owl_graph = Graph()
#     owl_graph.parse(data=EXAMPLE_OWL_TTL, format="turtle")

#     predictions = infereSubjectTypesFromProperties(rdf, owl_graph)

#     print(json.dumps(predictions, indent=4))


# # ENTITY_SIGNATURES = {str