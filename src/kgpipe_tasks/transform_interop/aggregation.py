# from kgflex.framework.kgflex import *
# from kgflex.resources.mainspec import *
from re import sub
from typing import List
from rdflib import Graph, URIRef, Literal, RDFS, RDF
from typing import Dict
from kgpipe.common import KgTask, Data, DataFormat, Registry
import json
import os
import logging
from pathlib import Path
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document

from scipy.sparse import data

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




@Registry.task(
    input_spec={"json1": DataFormat.TE_JSON, "json2": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Aggregate 2 TE_Document JSON files",
    category=["Aggregation"]
)
def aggregate2_te_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    __aggregate_x_te_json([inputs["json1"].path, inputs["json2"].path], outputs["output"].path)

@Registry.task(
    input_spec={"json1": DataFormat.TE_JSON, "json2": DataFormat.TE_JSON, "json3": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Aggregate 3 TE_Document JSON files",
    category=["Aggregation"]
)
def aggregate3_te_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    __aggregate_x_te_json([inputs["json1"].path, inputs["json2"].path, inputs["json3"].path], outputs["output"].path)


# def hash_uri(uri: str) -> str:
#     return hashlib.md5(uri.encode()).hexdigest()



# #######################################

# def aggregate_er_json(i: NameError, o:NamedData):
#     doc = ER_Document(**json.load(open(i["source"].path)))
#     graph = __generateRDF(doc)
#     graph.serialize(o["output"].path, format="nt")
#     print(f"RDF written to {o['output'].path}")

# #######################################

# def aggregate_2iejson(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     json1_path = inputs["json1"].path
#     json2_path = inputs["json2"].path

#     if os.path.isdir(json1_path) and os.path.isdir(json2_path):
#         # list files in each directory
#         json1_files = set(os.listdir(json1_path))
#         json2_files = set(os.listdir(json2_path))

#         # check for mismatches
#         if json1_files == json2_files:
#             os.makedirs(outputs["output"].path, exist_ok=True)
#             for file in json1_files:
#                 json1_doc = TE_Document(**json.load(open(os.path.join(json1_path, file))))
#                 json2_doc = TE_Document(**json.load(open(os.path.join(json2_path, file))))

#                 merged_doc = TE_Document()
#                 merged_doc.chains = json1_doc.chains + json2_doc.chains
#                 merged_doc.links = json1_doc.links + json2_doc.links
#                 merged_doc.triples = json1_doc.triples + json2_doc.triples

#                 with open(os.path.join(outputs["output"].path, file), "w") as f:
#                     f.write(merged_doc.model_dump_json())
#                 logger.info(f"Converted {file} to {os.path.join(outputs['output'].path, file)}")
#         else:
#             logger.error("File mismatch detected:")
#             logger.error("Files only in json1:", json1_files - json2_files)
#             logger.error("Files only in json2:", json2_files - json1_files)
#             logger.error("Common files in all:", json1_files & json2_files)
#             raise Exception("All input directories must contain the same file names")
#     else:
#         raise Exception("All inputs must be directories")

# aggregate_2iejson_task = KgTask(
#     name="aggregate_iejson_task",
#     input_spec={"json1": DataFormat.IE_JSON, "json2": DataFormat.IE_JSON},
#     output_spec={"output": DataFormat.IE_JSON},
#     function=aggregate_2iejson
# )

# # TODO
# def aggregate_3iejson(inputs: Dict[str, Data], outputs: Dict[str, Data]):

#     json1_path = inputs["json1"].path
#     json2_path = inputs["json2"].path
#     json3_path = inputs["json3"].path

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
        # raise Exception("All inputs must be directories")


def aggregate_3iejson_with_filter(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    json1_path = inputs["json1"].path
    json2_path = inputs["json2"].path
    json3_path = inputs["json3"].path

    def load_kg_uris_from_shades():
        """
        Loads the URIs of the entities in the current KG.
        """
        shade_file = "/home/marvin/project/data/current/shade_seed.json"
        with open(shade_file, "r") as f:
            return json.load(f)

    shade_dict = load_kg_uris_from_shades()
    reverse_shade_dict = {v: k for k, v in shade_dict.items()}
    kg_uris = set(shade_dict.values())
    
    # # Debug: Check for consistency
    # print(f"Debug: shade_dict has {len(shade_dict)} items")
    # print(f"Debug: reverse_shade_dict has {len(reverse_shade_dict)} items")
    # print(f"Debug: kg_uris has {len(kg_uris)} items")
    
    # # Check if there are any values in kg_uris that are not in reverse_shade_dict
    # missing_from_reverse = kg_uris - set(reverse_shade_dict.keys())
    # if missing_from_reverse:
    #     print(f"Debug: Found {len(missing_from_reverse)} values in kg_uris that are missing from reverse_shade_dict")
    #     print(f"Debug: First few missing values: {list(missing_from_reverse)[:5]}")
    
    # # Check if there are any values in reverse_shade_dict that are not in kg_uris
    # extra_in_reverse = set(reverse_shade_dict.keys()) - kg_uris
    # if extra_in_reverse:
    #     print(f"Debug: Found {len(extra_in_reverse)} values in reverse_shade_dict that are not in kg_uris")
    #     print(f"Debug: First few extra values: {list(extra_in_reverse)[:5]}")
    
    # # Test the logic: this should always be empty if the logic is correct
    # test_consistency = kg_uris - set(reverse_shade_dict.keys())
    # if test_consistency:
    #     print(f"ERROR: Logic inconsistency! {len(test_consistency)} values in kg_uris are missing from reverse_shade_dict")
    #     print(f"This should never happen if there are no duplicates in shade_dict.values()")
    #     print(f"Sample inconsistent values: {list(test_consistency)[:3]}")
    # else:
    #     print("Debug: Logic consistency check passed - all kg_uris values are in reverse_shade_dict")

    def filter_ie_doc(doc: TE_Document):
        """
        Removes links to entities that are not in the current KG.
        """

        # for uri in kg_uris:
            # print(uri)

        # Create a new list instead of modifying while iterating
        filtered_links = []
        for link in doc.links:
            if link.link_type == "entity":
                if link.mapping not in kg_uris:
                    # print(f"Removing entity link to {link.mapping} because it is not in the current KG")
                    continue  # Skip this link
                else:
                    tmp = link.mapping
                    try:
                        link.mapping = reverse_shade_dict[tmp]
                        # print(f"Replacing entity link {tmp} with {link.mapping}")
                    except KeyError:
                        print(f"KeyError: {tmp} not found in reverse_shade_dict, skipping")
                        continue  # Skip this link
            # elif link.link_type == "relation":
            #     if link.mapping not in kg_uris:
            #         print(f"Removing relation link to {link.mapping} because it is not in the current KG")
            #         continue  # Skip this link
            
            # Add the link to the filtered list (either it passed all checks or it's not an entity link)
            filtered_links.append(link)

        doc.links = filtered_links
        return doc


    if os.path.isdir(json1_path) and os.path.isdir(json2_path) and os.path.isdir(json3_path):
        # list files in each directory
        json1_files = set(os.listdir(json1_path))
        json2_files = set(os.listdir(json2_path))
        json3_files = set(os.listdir(json3_path))

        # check for mismatches
        if json1_files == json2_files == json3_files:
            os.makedirs(outputs["output"].path, exist_ok=True)
            for file in json1_files:
                json1_doc = TE_Document(**json.load(open(os.path.join(json1_path, file))))
                json2_doc = TE_Document(**json.load(open(os.path.join(json2_path, file))))
                json3_doc = TE_Document(**json.load(open(os.path.join(json3_path, file))))

                merged_doc = TE_Document()
                merged_doc.chains = json1_doc.chains + json2_doc.chains + json3_doc.chains
                merged_doc.links = json1_doc.links + json2_doc.links + json3_doc.links
                merged_doc.triples = json1_doc.triples + json2_doc.triples + json3_doc.triples

                merged_doc = filter_ie_doc(merged_doc)

                with open(os.path.join(outputs["output"].path, file), "w") as f:
                    f.write(merged_doc.model_dump_json())
                # print(f"Converted {file} to {os.path.join(outputs['output'].path, file)}")
        else:
            print("File mismatch detected:")
            print("Files only in json1:", json1_files - json2_files - json3_files)
            print("Files only in json2:", json2_files - json1_files - json3_files)
            print("Files only in json3:", json3_files - json1_files - json2_files)
            print("Common files in all:", json1_files & json2_files & json3_files)
            raise Exception("All input directories must contain the same file names")
    else:
        raise Exception("All inputs must be directories")

aggregate_3iejson_with_filter_task = KgTask(
    name="aggregate_iejson_with_filter_task",
    input_spec={"json1": DataFormat.TE_JSON, "json2": DataFormat.TE_JSON, "json3": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    function=aggregate_3iejson_with_filter
)

# # def aggregate_3iejson(i: NamedData, o:NamedData):

# #     json1_path = i["json1"].path
# #     json2_path = i["json2"].path
# #     json3_path = i["json3"].path

# #     if os.path.isdir(json1_path) and os.path.isdir(json2_path) and os.path.isdir(json3_path):
# #         # check each folder to contain the same file names
# #         json1_files = os.listdir(json1_path)
# #         json2_files = os.listdir(json2_path)
# #         json3_files = os.listdir(json3_path)

# #         if json1_files == json2_files == json3_files:
# #             os.makedirs(o["output"].path, exist_ok=True)
# #             for file in json1_files:
# #                 json1_doc = TE_Document(**json.load(open(os.path.join(json1_path, file))))
# #                 json2_doc = TE_Document(**json.load(open(os.path.join(json2_path, file))))
# #                 json3_doc = TE_Document(**json.load(open(os.path.join(json3_path, file))))

# #                 merged_doc = TE_Document()
# #                 merged_doc.chains = json1_doc.chains + json2_doc.chains + json3_doc.chains
# #                 merged_doc.links = json1_doc.links + json2_doc.links + json3_doc.links
# #                 merged_doc.triples = json1_doc.triples + json2_doc.triples + json3_doc.triples

# #                 with open(os.path.join(o["output"].path, file), "w") as f:
# #                     f.write(merged_doc.model_dump_json())
# #                 print(f"Converted {file} to {os.path.join(o['output'].path, file)}")
# #         else:
# #             print(json1_files)
# #             print("===")
# #             print(json2_files)
# #             print("===")
# #             print(json3_files)
# #             raise Exception("All inputs must contain the same file names")

# #     else:
# #         raise Exception("All inputs must be directories")



# aggregate_3iejson_task = KgTask(
#     name="aggregate_iejson_task",
#     input_spec={"json1": DataFormat.IE_JSON, "json2": DataFormat.IE_JSON, "json3": DataFormat.IE_JSON},
#     output_spec={"output": DataFormat.IE_JSON},
#     function=aggregate_3iejson
# )


# #######################################
# if __name__ == "__main__":
#     IN="/data/example.te.json"
#     OUT="/data/example.te.json.rdf"

#     doc = TE_Document(**json.load(open(IN)))
#     graph = __generateRDF(doc)
#     graph.serialize(destination=OUT, format='turtle')



# # # Variant 1 generic triples
# # triplets = extract_triplets("movie.json")
# # g = Graph()
# # for t in triplets:
# #     if str(t[2]).startswith("http"):
# #         g.add((URIRef(t[0]), URIRef(getPredicate(t[1])), URIRef(t[2])))
# #     else:
# #         g.add((URIRef(t[0]), URIRef(getPredicate(t[1])), Literal(t[2])))

# # # print(g.serialize(format="turtle"))

# # # # Variant 2 + link relations

# # # link relations
# # relations = [ t[1] for t in triplets]
# # links: List[TE_Pair] = link_relations(relations, "/home/marvin/papers/odibel/film_spec.ttl")

# # link_threshold = 0.5
# # link_map = {}
# # for l in links:
# #     if l.score > link_threshold:
# #         link_map[l.span] = l.mapping

# # g = Graph()
# # for t in triplets:
# #     if str(t[2]).startswith("http"):
# #         g.add((URIRef(t[0]), URIRef(link_map.get(t[1], getPredicate(t[1]))), URIRef(t[2])))
# #     else:
# #         g.add((URIRef(t[0]), URIRef(link_map.get(t[1], getPredicate(t[1]))), Literal(t[2])))