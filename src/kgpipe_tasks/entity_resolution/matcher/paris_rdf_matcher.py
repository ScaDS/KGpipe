"""
Paris RDF Matcher task implementation.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import os
import csv


from kgpipe.common import KgTask, DataFormat, Data, Registry
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Match, ER_Document

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.PARIS_CSV},  # Using Paris-specific format
    description="Paris entity matching using Docker container"
)
def paris_entity_matching(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    Paris entity matching task that runs in a Docker container.
    
    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """
    # print(f"Running Paris entity matching with inputs: {inputs}")
    
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    # Extract input paths
    source_path = remap_data_path_for_container(inputs["source"], host_to_container)
    target_path = remap_data_path_for_container(inputs["kg"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)
    
    # Ensure output directory exists
    outputs["output"].path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all data for Docker volume bindings

    # Create Docker client with proper volume bindings
    client = docker_client(
        image="kgt/paris:latest",
        # command=["ls", "-la"],
        command=["bash", "paris.sh", 
                 str(source_path.path),
                 str(target_path.path),
                 str(output_path.path)],
        volumes=volumes,
    )
    
    # Execute the container
    result = client()
    print(f"Paris entity matching completed: {result}")



    
PREFIX_MAP = {
    "dbp": "http://dbpedia.org/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf" : "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd" : "http://www.w3.org/2001/XMLSchema#",
    "schema" : "http://schema.org/",
    "dbo": "http://dbpedia.org/ontology/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

def resolvePrefixedUri(uri):
    if not uri.startswith("http://") and not uri.startswith("https://"):
        prefix, suffix = uri.split(":", 1)
        # try:
        prefix = PREFIX_MAP[prefix]
        # except Exception as e:
        #     print(f"Unknown prefix: {prefix} for {uri}")
        #     raise Exception(f"Unknown prefix: {prefix} for {uri}")
        return prefix + suffix
    else:
        return uri


@Registry.task(
    input_spec={"input": DataFormat.PARIS_CSV},
    output_spec={"output": DataFormat.ER_JSON},
    description="Convert Paris CSV output to standard RDF matching format"
)
def paris_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    Convert Paris CSV output to standard RDF matching format.
    
    Args:
        inputs: Dictionary mapping input names to Data objects (Paris CSV)
        outputs: Dictionary mapping output names to Data objects (RDF)
    """
    print(f"Converting Paris CSV to matching format with inputs: {inputs}")

    input_path = inputs["input"].path
    output_path = outputs["output"].path
    
    Path(output_path).touch()
    files = [str(f) for f in os.listdir(input_path)]

    iteration_ids = [ int(f.split("_")[0]) for f in files if f.endswith(".tsv") ]

    iteration_ids.sort()

    last_eqv_it = iteration_ids[-1]

    def getEqvFileName(id): return f"{id}_eqv.tsv"
    def getRelFileNames(id): return [f"{id}_superrelations1.tsv",f"{id}_superrelations2.tsv"]

    def check_file_exists(last_eqv_it):
        try:
            return os.stat(os.path.join(input_path, getEqvFileName(last_eqv_it))).st_size > 0
        except FileNotFoundError:
            return -1

    while 0 == check_file_exists(last_eqv_it) :
        last_eqv_it -= 1

    last_relation_it = last_eqv_it - 1

    def extract_matches(file,id_type):
        with open(file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if len(row) == 3:
                    er_match = ER_Match(
                        id_1=resolvePrefixedUri(row[0]),
                        id_2=resolvePrefixedUri(row[1]),
                        score=float(row[2]),
                        id_type=id_type
                    )
                    matches.append(er_match)

    matches = []

    if last_eqv_it == -1:
        doc = ER_Document(matches=[])
        with open(outputs["output"].path, 'w', encoding='utf-8') as jsonfile:
            jsonfile.write(doc.model_dump_json())
    else:
        eqv_file = getEqvFileName(last_eqv_it)
        rel_files = getRelFileNames(last_relation_it)

        extract_matches(os.path.join(input_path,eqv_file),"entity")
        [ extract_matches(os.path.join(input_path,f), "relation") for f in rel_files ]

        doc = ER_Document(matches=matches)

        with open(outputs["output"].path, 'w', encoding='utf-8') as jsonfile:
            jsonfile.write(doc.model_dump_json())

