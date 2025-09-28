from typing import Dict, List
from kgpipe.common import Data, DataFormat, Registry
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document, TE_Triple, TE_Pair, TE_Chains
from kgpipe_tasks.common.ontology import Ontology, OntologyUtil

from rdflib import Graph, URIRef, Literal, RDF, RDFS, XSD
from kgpipe_tasks.common.benchutils import hash_uri

import json
import os
from pathlib import Path

# @Registry.task(
#     input_spec={"source": DataFormat.JSON},
#     output_spec={"output": DataFormat.RDF_NTRIPLES},
#     description="Construct RDF graph from TE_Document",
#     category=["Construction"]
# )
# def construct_rdf_from_te_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     pass

# @Registry.task(
#     input_spec={"source": DataFormat.JSON},
#     output_spec={"output": DataFormat.IE_JSON},
#     description="Construct TE_Document from JSON",
#     category=["Construction"]
# )
# def construct_te_document_from_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     pass

ENTITY_LINK_THRESHOLD = 0.5
PREDICATE_LINK_THRESHOLD = 0.3

##########################

def generatePredicate(surface_form, namespace):
    return URIRef(namespace + surface_form.replace(" ", "_"))

def __generateRDF(doc: TE_Document, ontology: Ontology, generic: bool = False, namespace: str = "http://kg.org/text/"):
    """
    A processing node, part of a pipeline
    collects information from extractors, linkers, and resolvers and then it produces the final triples
    """


    def process_chains(triples, chains: List[TE_Chains]):
        new_triples = triples
        chain_dict = {}
        for chain in chains:
            for alias in chain.aliases:
                chain_dict[alias.surface_form] = chain.main
        if len(chain_dict) > 0:
            # TODO check if chain_dict should be a dict of TE_SPANS to avoid merging
            for triple in new_triples:
                if triple.subject.surface_form in chain_dict:
                    triple.subject.surface_form = chain_dict[triple.subject.surface_form]
                if triple.object.surface_form in chain_dict:
                    triple.object.surface_form = chain_dict[triple.object.surface_form]
        return new_triples

    def process_links(triples, links: List[TE_Pair]):
        new_triples = triples
        try:
            if len(links) > 0:
                so_spans = {}
                p_spans = {}
                for triple in triples:
                    # Add Subject spans
                    if triple.subject.surface_form.lower().startswith("http://"):
                        pass
                    elif triple.subject.surface_form.lower() not in so_spans:
                        so_spans[triple.subject.surface_form.lower()] = [triple.subject]
                    else:
                        so_spans[triple.subject.surface_form.lower()].append(triple.subject)
                    # Add object spans
                    if triple.object.surface_form.lower().startswith("http://"):
                        pass
                    elif triple.object.surface_form.lower() not in so_spans:
                        so_spans[triple.object.surface_form.lower()] = [triple.object]
                    else:
                        so_spans[triple.object.surface_form.lower()].append(triple.object)
                    # add predicate spans
                    if triple.predicate.surface_form.lower().startswith("http://"):
                        pass
                    elif triple.predicate.surface_form.lower() not in p_spans:
                        p_spans[triple.predicate.surface_form.lower()] = [triple.predicate]
                    else:
                        p_spans[triple.predicate.surface_form.lower()].append(triple.predicate)
                for link in links:
                    if link.link_type == 'entity':
                        spans = so_spans
                        if link.score < ENTITY_LINK_THRESHOLD:
                            continue
                    else:
                        spans = p_spans
                        if link.score < PREDICATE_LINK_THRESHOLD:
                            continue
                    if link.span and link.span.lower() in spans:
                        for span in spans[link.span.lower()]:
                            span.mapping = link.mapping
                            # span.surface_form = link.mapping
        except Exception as exp:
            raise exp
        finally:
            return new_triples
        
    
    triples: List[TE_Triple] = doc.triples
    links: List[TE_Pair] = doc.links
    chains: List[TE_Chains] = doc.chains

    dereferenced_tiples = process_chains(triples, chains)
    linked_triples: List[TE_Triple] = process_links(dereferenced_tiples, links)
    finalGraph = Graph()

    for triple in linked_triples:
        subject = None
        if triple.subject.mapping:
            subject = URIRef(triple.subject.mapping)
        # else:
        #     subject = triple.subject.surface_form

        predicate = None
        if triple.predicate.mapping:
            predicate = URIRef(triple.predicate.mapping)
        elif generic:
            predicate = generatePredicate(triple.predicate.surface_form, namespace)

        object = None
        # TODO if predicate is a datatype or object property
        if triple.object.mapping:
            object = URIRef(triple.object.mapping)
        # else:
        #     object = Literal(triple.object.surface_form)
        # if(subject and predicate and object):
        #     finalGraph.add((subject, predicate, object))

        # new entities
        if(predicate):
            # print(f"new subject: {subject} {triple.subject.surface_form}")

            domain, range = ontology.get_domain_range(str(predicate))
            isObjectProperty = True if range and range.startswith("http://kg.org") else False
            # print(f"predicate: {predicate}, domain: {domain}, range: {range}, isObjectProperty: {isObjectProperty}")
            # print(f"predicate: {predicate}, domain: {domain}, range: {range}")

            if subject and subject.startswith("http://dbpedia.org"): # TODO workaround for dbpedia...
                finalGraph.add((subject, RDFS.label, Literal(triple.subject.surface_form)))


            if not subject and triple.subject.surface_form:
                subject = URIRef(namespace+hash_uri(triple.subject.surface_form))
                finalGraph.add((subject, RDFS.label, Literal(triple.subject.surface_form)))
                print(f"new subject: {subject} {triple.subject.surface_form}")
            else:
                print(f"subject: {subject} {triple.subject.surface_form}")

            if domain and subject:
                finalGraph.add((subject, RDF.type, URIRef(domain)))

            if not object and triple.object.surface_form:
                if isObjectProperty:
                    object = URIRef(namespace+hash_uri(triple.object.surface_form))
                    finalGraph.add((object, RDFS.label, Literal(triple.object.surface_form)))
                    if range:
                        finalGraph.add((object, RDF.type, URIRef(range)))
                else:
                    datatype = range if range else str(XSD.string)
                    object = Literal(triple.object.surface_form, datatype=datatype)
            else:
                if not isObjectProperty:
                    datatype = range if range else str(XSD.string)
                    object = Literal(triple.object.surface_form, datatype=datatype)

        if(subject and predicate and object):
            finalGraph.add((subject, predicate, object))
        
    return finalGraph


# # def generate_generic_rdf(i:NamedData, od: NamedData):
# #     dir_or_file = i["source"].path
# #     graph = Graph()
# #     if os.path.isdir(dir_or_file):
# #         for file in os.listdir(dir_or_file):
# #             json_data = json.load(open(os.path.join(dir_or_file, file)))
# #             doc = TE_Document(**json_data)
# #             for s, p, o in __generateRDF(doc, generic=True):
# #                 graph.add(triple=(s, p, o))
# #     else:
# #         doc = TE_Document(**json.load(dir_or_file))
# #         graph = __generateRDF(doc, generic=True)

# #     graph.serialize(od["output"].path, format="nt")
# #     print(f"RDF written to {od['output'].path}")

# # generate_generic_rdf_task = KgTask(
# #     name="generate_generic_rdf_task",
# #     inputDict={"source": IE_JSON},
# #     outputDict={"output": RDF_NT},
# #     function=generate_generic_rdf
# # )

# def generate_rdf(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     dir_or_file = inputs["source"].path
#     ontology = OntologyUtil.load_ontology_from_file("/home/marvin/project/data/current/ontology.ttl")
#     graph = Graph()
#     if os.path.isdir(dir_or_file):
#         for file in os.listdir(dir_or_file):
#             json_data = json.load(open(os.path.join(dir_or_file, file)))
#             doc = TE_Document(**json_data)
#             for s, p, o in __generateRDF(doc):
#                 graph.add(triple=(s, p, o))
#     else:
#         doc = TE_Document(**json.load(open(dir_or_file)))
#         graph = __generateRDF(doc, ontology=ontology)

#     graph.serialize(outputs["output"].path, format="nt")
#     print(f"RDF written to {outputs['output'].path}")

# def generate_rdf_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     dir_or_file = inputs["source"].path
#     ontology = OntologyUtil.load_ontology_from_file("/home/marvin/project/data/current/ontology.ttl")
#     graph = Graph()
#     if os.path.isdir(dir_or_file):
#         for file in os.listdir(dir_or_file):
#             json_data = json.load(open(os.path.join(dir_or_file, file)))
#             doc = TE_Document(**json_data)
#             for s, p, o in __generateRDF(doc, ontology=ontology, namespace="http://kg.org/json/"):
#                 graph.add(triple=(s, p, o))
#     else:
#         doc = TE_Document(**json.load(open(dir_or_file)))
#         graph = __generateRDF(doc, ontology=ontology, namespace="http://kg.org/json/")
#     graph.serialize(outputs["output"].path, format="nt")
#     print(f"RDF written to {outputs['output'].path}")

# def generate_rdf_text(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     dir_or_file = inputs["source"].path
#     ontology = OntologyUtil.load_ontology_from_file("/home/marvin/project/data/current/ontology.ttl")
#     graph = Graph()
#     if os.path.isdir(dir_or_file):
#         for file in os.listdir(dir_or_file):
#             json_data = json.load(open(os.path.join(dir_or_file, file)))
#             doc = TE_Document(**json_data)
#             for s, p, o in __generateRDF(doc, ontology=ontology, namespace="http://kg.org/text/"):
#                 graph.add(triple=(s, p, o))
#     else:
#         doc = TE_Document(**json.load(open(dir_or_file)))
#         graph = __generateRDF(doc, ontology=ontology, namespace="http://kg.org/text/")

#     graph.serialize(outputs["output"].path, format="nt")
#     print(f"RDF written to {outputs['output'].path}")

# generate_rdf_task = KgTask(
#     name="generate_rdf_task",
#     input_spec={"source": DataFormat.IE_JSON},
#     output_spec={"output": DataFormat.RDF_NTRIPLES},
#     function=generate_rdf
# )

# generate_rdf_json_task = KgTask(
#     name="generate_rdf_json_task",
#     input_spec={"source": DataFormat.IE_JSON},
#     output_spec={"output": DataFormat.RDF_NTRIPLES},
#     function=generate_rdf_json
# )

# generate_rdf_text_task = KgTask(
#     name="generate_rdf_text_task",
#     input_spec={"source": DataFormat.IE_JSON},
#     output_spec={"output": DataFormat.RDF_NTRIPLES},
#     function=generate_rdf_text
# )



def generate_rdf(inputs: Dict[str, Data], outputs: Dict[str, Data], ontology: Ontology, generic: bool):
    dir_or_file = inputs["source"].path
    graph = Graph()
    if os.path.isdir(dir_or_file):
        for file in os.listdir(dir_or_file):
            json_data = json.load(open(os.path.join(dir_or_file, file)))
            doc = TE_Document(**json_data)
            for s, p, o in __generateRDF(doc, ontology, generic=generic):
                graph.add(triple=(s, p, o))
    else:
        doc = TE_Document(**json.load(open(dir_or_file)))
        graph = __generateRDF(doc, ontology, generic=generic)

    graph.serialize(outputs["output"].path, format="nt")
    print(f"RDF written to {outputs['output'].path}")

# generate_rdf_generic_task = KgTask(
#     name="generate_rdf_generic_task",
#     input_spec={"source": DataFormat.IE_JSON},
#     output_spec={"output": DataFormat.RDF_NTRIPLES},
#     function=generate_rdf_generic
# )

@Registry.task(
    input_spec={"source": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Construct RDF graph from TE_Document",
    category=["Construction"]
)
def construct_rdf_from_te_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):

    ontology_path = os.environ.get("ONTOLOGY_PATH", "false")
    if ontology_path == "false":
        raise ValueError("ONTOLOGY_PATH is not set")

    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))

    generate_rdf(inputs, outputs, ontology, generic=False)
