import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from kgcore.api.ontology import Ontology, OntologyUtil
from kgpipe.common import Data, DataFormat, KgTask
from kgpipe.common.model.configuration import ConfigurationDefinition
from kgpipe_tasks.common.benchutils import hash_uri
from kgpipe_tasks.transform_interop.exchange.text_extraction import (
    TE_Chains,
    TE_Document,
    TE_Pair,
    TE_Triple,
)
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

logger = logging.getLogger(__name__)

def __aggregate_x_te_json(input_paths: List[Path], output_path: Path):

    if len(input_paths) == 0:
        raise Exception("No input paths provided")
    if not all(os.path.exists(path) for path in input_paths):
        raise Exception("All input paths must exist")

    path_is_dir_list = [os.path.isdir(path) for path in input_paths]
    if all(path_is_dir_list):
        os.makedirs(output_path, exist_ok=True)
        for file in os.listdir(input_paths[0]):
            sub_file_paths = [Path(os.path.join(path, file)) for path in input_paths]
            file_exists = [os.path.exists(path) for path in sub_file_paths]
            if all(file_exists):
                __aggregate_x_te_json(sub_file_paths, Path(os.path.join(output_path, file))) 
            else:
                logger.warning(f"File {file} does not exist in all input paths")
                filtered_sub_file_paths = [path for path in sub_file_paths if os.path.exists(path)]
                __aggregate_x_te_json(filtered_sub_file_paths, Path(os.path.join(output_path, file))) 
    elif not all(path_is_dir_list):
        merged_doc = TE_Document()
        for file in input_paths:
            doc = TE_Document(**json.load(open(file)))
            merged_doc.chains += doc.chains
            merged_doc.links += doc.links
            merged_doc.triples += doc.triples
        with open(output_path, "w") as f:
            f.write(merged_doc.model_dump_json())
        logger.info(f"Aggregated {", ".join([str(path) for path in input_paths])} to {output_path}")
    else:
        raise Exception("All inputs must be either directories or files")


# @Registry.task(
#     input_spec={"json1": DataFormat.TE_JSON, "json2": DataFormat.TE_JSON},
#     output_spec={"output": DataFormat.TE_JSON},
#     description="Aggregate 2 TE_Document JSON files",
#     category=["Aggregation"]
# )
# def aggregate2_te_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     __aggregate_x_te_json([inputs["json1"].path, inputs["json2"].path], outputs["output"].path)


def aggregate3_text_tasks_task_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    __aggregate_x_te_json([inputs["json1"].path, inputs["json2"].path, inputs["json3"].path], outputs["output"].path)

aggregate_text_tasks_task = KgTask(
    name="aggregate3_text_tasks_task",
    input_spec={"json1": DataFormat.TE_JSON, "json2": DataFormat.TE_JSON, "json3": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    function=aggregate3_text_tasks_task_function
)


def generatePredicate(surface_form, namespace):
    return URIRef(namespace + surface_form.replace(" ", "_"))

def __generateRDF(doc: TE_Document, ontology: Ontology, newP: bool = False, newE: bool = False, namespace: str = "http://kg.org/text/"):
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
                    else:
                        spans = p_spans
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
        elif newP:
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


            if not subject and triple.subject.surface_form and newE:
                subject = URIRef(namespace+hash_uri(triple.subject.surface_form))
                finalGraph.add((subject, RDFS.label, Literal(triple.subject.surface_form)))
                print(f"new subject: {subject} {triple.subject.surface_form}")
            else:
                print(f"subject: {subject} {triple.subject.surface_form}")

            if domain and subject:
                finalGraph.add((subject, RDF.type, URIRef(domain)))

            if not object and triple.object.surface_form and newE:
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


def generate_rdf(inputs: Dict[str, Data], outputs: Dict[str, Data], ontology: Ontology, newP: bool, newE: bool):
    dir_or_file = inputs["source"].path
    graph = Graph()
    if os.path.isdir(dir_or_file):
        for file in os.listdir(dir_or_file):
            json_data = json.load(open(os.path.join(dir_or_file, file)))
            doc = TE_Document(**json_data)
            for s, p, o in __generateRDF(doc, ontology, newP=newP, newE=newE):
                graph.add(triple=(s, p, o))
    else:
        doc = TE_Document(**json.load(open(dir_or_file)))
        graph = __generateRDF(doc, ontology, newP=newP, newE=newE)

    graph.serialize(outputs["output"].path, format="nt")
    print(f"RDF written to {outputs['output'].path}")


def generate_rdf_from_text_results_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):

    ontology_path = os.environ.get("ONTOLOGY_PATH", "false")
    if ontology_path == "false":
        raise ValueError("ONTOLOGY_PATH is not set")

    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))

    generate_rdf(inputs, outputs, ontology, newP=False, newE=True)


generate_rdf_from_text_results_task = KgTask(
    name="construct_rdf_from_text_tasks_task",
    input_spec={"source": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    function=generate_rdf_from_text_results_function
)


# ------------------------------------------------------------


# def aggregate_3iejson_with_filter(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     json1_path = inputs["json1"].path
#     json2_path = inputs["json2"].path
#     json3_path = inputs["json3"].path

#     def load_kg_uris_from_shades():
#         """
#         Loads the URIs of the entities in the current KG.
#         """
#         shade_file = "/home/marvin/project/data/current/shade_seed.json"
#         with open(shade_file, "r") as f:
#             return json.load(f)

#     shade_dict = load_kg_uris_from_shades()
#     reverse_shade_dict = {v: k for k, v in shade_dict.items()}
#     kg_uris = set(shade_dict.values())
    

#     def filter_ie_doc(doc: TE_Document):
#         """
#         Removes links to entities that are not in the current KG.
#         """

#         # for uri in kg_uris:
#             # print(uri)

#         # Create a new list instead of modifying while iterating
#         filtered_links = []
#         for link in doc.links:
#             if link.link_type == "entity":
#                 if link.mapping not in kg_uris:
#                     # print(f"Removing entity link to {link.mapping} because it is not in the current KG")
#                     continue  # Skip this link
#                 else:
#                     tmp = link.mapping
#                     try:
#                         link.mapping = reverse_shade_dict[tmp]
#                         # print(f"Replacing entity link {tmp} with {link.mapping}")
#                     except KeyError:
#                         print(f"KeyError: {tmp} not found in reverse_shade_dict, skipping")
#                         continue  # Skip this link
#             # elif link.link_type == "relation":
#             #     if link.mapping not in kg_uris:
#             #         print(f"Removing relation link to {link.mapping} because it is not in the current KG")
#             #         continue  # Skip this link
            
#             # Add the link to the filtered list (either it passed all checks or it's not an entity link)
#             filtered_links.append(link)

#         doc.links = filtered_links
#         return doc


#     if os.path.isdir(json1_path) and os.path.isdir(json2_path) and os.path.isdir(json3_path):
#         # list files in each directory
#         json1_files = set(os.listdir(json1_path))
#         json2_files = set(os.listdir(json2_path))
#         json3_files = set(os.listdir(json3_path))

#         # check for mismatches
#         if json1_files == json2_files == json3_files:
#             os.makedirs(outputs["output"].path, exist_ok=True)
#             for file in json1_files:
#                 json1_doc = TE_Document(**json.load(open(os.path.join(json1_path, file))))
#                 json2_doc = TE_Document(**json.load(open(os.path.join(json2_path, file))))
#                 json3_doc = TE_Document(**json.load(open(os.path.join(json3_path, file))))

#                 merged_doc = TE_Document()
#                 merged_doc.chains = json1_doc.chains + json2_doc.chains + json3_doc.chains
#                 merged_doc.links = json1_doc.links + json2_doc.links + json3_doc.links
#                 merged_doc.triples = json1_doc.triples + json2_doc.triples + json3_doc.triples

#                 merged_doc = filter_ie_doc(merged_doc)

#                 with open(os.path.join(outputs["output"].path, file), "w") as f:
#                     f.write(merged_doc.model_dump_json())
#                 # print(f"Converted {file} to {os.path.join(outputs['output'].path, file)}")
#         else:
#             print("File mismatch detected:")
#             print("Files only in json1:", json1_files - json2_files - json3_files)
#             print("Files only in json2:", json2_files - json1_files - json3_files)
#             print("Files only in json3:", json3_files - json1_files - json2_files)
#             print("Common files in all:", json1_files & json2_files & json3_files)
#             raise Exception("All input directories must contain the same file names")
#     else:
#         raise Exception("All inputs must be directories")

# aggregate_3iejson_with_filter_task = KgTask(
#     name="aggregate_iejson_with_filter_task",
#     input_spec={"json1": DataFormat.TE_JSON, "json2": DataFormat.TE_JSON, "json3": DataFormat.TE_JSON},
#     output_spec={"output": DataFormat.TE_JSON},
#     function=aggregate_3iejson_with_filter
# )

