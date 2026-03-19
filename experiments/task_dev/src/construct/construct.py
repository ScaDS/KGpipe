from kgpipe.common import Registry, Data, DataFormat
from kgcore.model.ontology import Ontology, OntologyUtil
import json
from kgpipe_llm.common.api_utils import get_token_count
from kgpipe_llm.common.core import get_client_from_env, LLMClient
from pydantic import BaseModel
from typing import Dict, List, Generator, Optional
import typing
import os
from rdflib import Graph, URIRef, Literal, RDFS, RDF, OWL, XSD, SKOS, DC, DCAT, FOAF
# from .schema_models import JsonLDDocument, JSON_LD_SCHEMA_DICT
# from .json_ld import JSON_SCHEMA_DICT_v2
from kgpipe_llm.common.snippets import generate_ontology_snippet_v2
from pathlib import Path
from tqdm import tqdm

import hashlib

# Config

TOKEN_LIMIT = int(os.environ.get("TOKEN_LIMIT", 8192)) #*16384))

# === Response Schemas ===

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    object_type: typing.Literal["uri", "literal"]
    object_datatype: Optional[str]

class Triples(BaseModel):
    triples: List[Triple]

# === Util Functions ===

def __mint_uri(label: str) -> URIRef:

    return URIRef(f"http://kgflex.org/llm/{hashlib.sha256(label.encode()).hexdigest()}")

def __write_response_plain(response, output_path):
    if isinstance(response, dict):
        with open(output_path, "w") as f:
            f.write(json.dumps(response, indent=4))
    else:
        with open(output_path, "w") as f:
            f.write(json.dumps(response.model_dump(), indent=4))

def __init_graph_with_prefixes() -> Graph:
    graph = Graph()
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("owl", OWL)
    graph.bind("xsd", XSD)
    graph.bind("skos", SKOS)
    graph.bind("dc", DC)
    graph.bind("dcat", DCAT)
    graph.bind("foaf", FOAF)
    graph.bind("schema", "http://schema.org/")
    graph.bind("dbr", "http://dbpedia.org/resource/")
    graph.bind("dbo", "http://dbpedia.org/ontology/")
    graph.bind("dbp", "http://dbpedia.org/property/")
    graph.bind("kgo", "http://kg.org/ontology/")
    graph.bind("kgr", "http://kg.org/resource/")
    return graph

def __parse_rdf_file_to_graph(file_path: Path) -> Graph:
    
    graph = __init_graph_with_prefixes()
    graph.parse(file_path, format="nt")

    return graph

def __parse_triples_to_graph(response) -> Graph:
    print("__parse_triples_to_graph")
    graph = Graph()
    if isinstance(response, dict):
        is_dict = True
        triples = response["triples"]
    else:
        is_dict = False
        triples = response.triples
    for triple in triples:

        if is_dict:
            subject = triple.get("subject", None)
            predicate = triple.get("predicate", None)
            object = triple.get("object", None)
            object_datatype = triple.get("object_datatype", None)
            object_type = triple.get("object_type", "uri")
        else:
            subject = triple.subject
            predicate = triple.predicate
            object = triple.object
            object_datatype = triple.object_datatype
            object_type = triple.object_type

        if subject and subject.startswith("http"):
            s = URIRef(subject)
        elif subject:
            s = __mint_uri(subject)
            graph.add((s, RDFS.label, Literal(subject)))
        else:
            continue
        if predicate and predicate.startswith("http"):
            p = URIRef(predicate)
        elif predicate and ("type" in predicate or "a" == predicate): # TODO hack
            p = RDF.type
        elif predicate and ("label" in predicate): # TODO hack
            p = RDFS.label
        elif predicate:
            p = __mint_uri(predicate)
            graph.add((p, RDFS.label, Literal(predicate)))
        else:
            continue
        if object and object.startswith("http"):
            o = URIRef(object)
        elif object and object_type == "uri":
            o = __mint_uri(object)
            graph.add((o, RDFS.label, Literal(object)))
        elif object and object_type == "literal":
            if object_datatype and object_datatype.startswith("xsd:"):
                o = Literal(object, datatype=object_datatype.replace("xsd:", "http://www.w3.org/2001/XMLSchema#"))
            elif object_datatype and object_datatype.startswith("http"):
                o = Literal(object, datatype=object_datatype)
            else:
                o = Literal(object)
        else:
            continue
        graph.add((s, p, o))
    return graph

def __parse_json_ld_to_graph(response: Dict | BaseModel) -> Graph:
    print("__parse_json_ld_to_graph")
    graph = Graph()
    try:
        if isinstance(response, dict):
            graph.parse(data=json.dumps(response, indent=4), format="json-ld")
        else:
            graph.parse(data=response.model_dump_json(indent=4), format="json-ld")
    except Exception as e:
        print(f"Error parsing JSON-LD: {e}")
        # print(json.dumps(response, indent=4))
        # raise e
    return graph

def __parse_response_to_graph(response: Dict | BaseModel, schema_class) -> Graph:
    print("__parse_response_to_graph")
    # if has attribute triples, then parse as triples
    if type(schema_class) == type(Triples):
        graph = __parse_triples_to_graph(response)
    else:
        graph = __parse_json_ld_to_graph(response)
    return graph


def input_splitting_prompt_generator(
    input_path: Path, 
    base_prompt: str, 
    prompt_mappings: Dict[str, str], 
    token_limit: int = 16384) -> Generator[str, None, None]:
    """
    For the given input, based on the context, and the token limit,
    splits the input into parts, and returns the parts.
    """

    def part_for_sub_file(file_path: Path) -> Dict | str:
        if file_path.suffix == ".json":
            json_data = open(file_path, "r").read()
            return json.loads(json_data)
        elif file_path.suffix == ".txt":
            text = open(file_path, "r").read()
            return text
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def parts_for_single_file(file_path: Path, token_limit: int) -> Generator[str, None, None]:
        
        if file_path.suffix in [".nt", ".ttl"]:
            graph = __parse_rdf_file_to_graph(file_path)

            left_entities = list(set(graph.subjects(unique=True)))
            part_graph = __init_graph_with_prefixes()
            part_count = 0

            for entity in left_entities:
                for s, p, o in graph.triples((entity, None, None)):
                    part_graph.add((s, p, o))
                    current_prompt = base_prompt + "\n".join(part_graph.serialize(format="turtle"))
                    token_count = get_token_count(current_prompt)
                    if token_count > (token_limit * 0.8):
                        part_graph = part_graph.remove((None, None, None))
                        part_count += 1
                        print(f"Part {part_count} token count: {token_count}")
                        yield current_prompt
            if len(part_graph) > 0:
                current_prompt = base_prompt + "\n".join(part_graph.serialize(format="turtle"))
                yield current_prompt
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # if is directory, then split the input into parts
    if input_path.is_dir():
        current_parts = []
        for file in input_path.iterdir():
            if file.is_file():
                part = part_for_sub_file(file)
                current_parts.append(part)
                current_prompt = base_prompt + "\n"+json.dumps(current_parts, indent=4, default=str)
                if get_token_count(current_prompt) > (token_limit * 0.8):
                    yield current_prompt
                    current_parts = []
        if current_parts:
            yield current_prompt
    else:
        for part in parts_for_single_file(input_path, token_limit):
            yield part


def execute_prompt_for_input(
    input: Dict[str, Data], 
    output: Dict[str, Data], 
    prompt: str,
    system_prompt: str,
    llm: LLMClient,
    llm_schema):
    graph = Graph()

    parts = input_splitting_prompt_generator(input["input"].path, prompt, {"input": "INPUT"}, TOKEN_LIMIT)

    prompt_log_file = open(output["output"].path.as_posix() + "_prompt.txt", "w")
    raw_response_file = output["output"].path.as_posix() + "_response.json"

    for part_count, part in enumerate(tqdm(list(parts))):

        prompt_log_file.write(f"=== PART {part_count} ===\n\n")
        prompt_log_file.write(part)
        prompt_log_file.write("\n\n")
        prompt_log_file.flush()

        if get_token_count(part) > TOKEN_LIMIT:
            raise ValueError(f"Prompt is too long max {TOKEN_LIMIT} tokens, got {get_token_count(part)} tokens")
        
        triples = llm.send_prompt(part, llm_schema, system_prompt=system_prompt)
        if triples:
            __write_response_plain(triples, raw_response_file)
            for s, p, o in __parse_response_to_graph(triples, llm_schema):
                graph.add((s, p, o))

    graph.serialize(destination=output["output"].path, format="nt")


# === Tasks ===

@Registry.task(
    description="Construct a knowledge graph from any input.",
    input_spec={"input": DataFormat.ANY},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["Construction", "LLM"]
)
def construct_from_any_input_as_triples(input: Dict[str, Data], output: Dict[str, Data]):
    """
    Construct a knowledge graph from any input.
    """
    llm = get_client_from_env()
    
    json_string = open(input["input"].path, "r").read()

    prompt = """
    Construct a knowledge graph from the provided INPUT. 
    For each entity generate at least a type and a label triple.
    Return the triples in the following format:
    [
        {{ "subject": "...", "predicate": "...", "object": "..." }},
        ...
    ]

    === INPUT ===

    {input}
    """.format(input=json_string)

    system_prompt = "You are a KG engineer. Generating subject-predicate-object triples."
    llm = get_client_from_env()
    execute_prompt_for_input(input, output, prompt, system_prompt, llm, Triples)
    
@Registry.task(
    description="Construct a knowledge graph from any input.",
    input_spec={"input": DataFormat.ANY},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["Construction", "LLM"]
)
def construct_from_any_input_with_ontology_as_triples(input: Dict[str, Data], output: Dict[str, Data]):
    """
    Construct a knowledge graph from any input with a given ontology.
    """
    
    ontology_path = os.environ.get("ONTOLOGY_PATH")
    if not ontology_path:
        raise ValueError("ONTOLOGY_PATH is not set")
    
    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    ontology_snippet = generate_ontology_snippet_v2(ontology)

    prompt = """
Construct a knowledge graph from the provided INPUT and ONTOLOGY.
For each extracted entity generate at least a type and a label triple.
Return the triples in the following format:
[
    {{ "subject": "...", "predicate": "...", "object": "..." }},
    ...
]

=== ONTOLOGY ===

{ontology_snippet}

=== INPUT ===
""".format(ontology_snippet=ontology_snippet)

    system_prompt = "You are a KG engineer. Generating subject-predicate-object triples."
    llm = get_client_from_env()
    execute_prompt_for_input(input, output, prompt, system_prompt, llm, Triples)


@Registry.task(
    description="Construct a knowledge graph from any input with a knowledge graph.",
    input_spec={"input": DataFormat.ANY, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["Construction", "LLM"]
)
def construct_from_any_input_with_kg_as_triples(input: Dict[str, Data], output: Dict[str, Data]):
    """
    Construct a knowledge graph from any input with current knowledge graph.
    The current knowledge graph is given as a TTL file.
    The input is given as a JSON file.
    The output is expected to be in JSON-LD format.
    """
    
    kg_path = input["target"].path
    graph = Graph()
    graph.parse(kg_path, format="nt")

    kg_data = graph.serialize(format="turtle")
    input_data = open(input["input"].path, "r").read()

    prompt = """
    Construct a knowledge graph in JSON-LD from the INPUT,
    for the given current TARGET knowledge graph RDF Turtle.
    Only use the TARGET KG predicates and classes.
    Try to reuse exisitng entity URIs, if there is a possible match, else new URI.
    Return only the triples in the following format:
    [
        {{ "subject": "...", "predicate": "...", "object": "..." }},
        ...
    ]

    === TARGET ===

    {target}

    === INPUT ===

    {input}
    """.format(input=input_data, target=kg_data)

    system_prompt = "You are a KG engineer. Generating subject-predicate-object triples."
    llm = get_client_from_env()
    execute_prompt_for_input(input, output, prompt, system_prompt, llm, Triples)


@Registry.task(
    description="Construct a knowledge graph from any input with a knowledge graph.",
    input_spec={"input": DataFormat.ANY, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_TTL},
    category=["Construction", "LLM"]
)
def construct_from_any_input_with_kg_and_ontology_as_triples(input: Dict[str, Data], output: Dict[str, Data]):
    raise NotImplementedError("Not implemented yet")

@Registry.task(
    description="Construct a knowledge graph from any input with a knowledge graph.",
    input_spec={"input": DataFormat.ANY},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["Construction", "LLM"]
)
def construct_from_any_input_as_json_ld(input: Dict[str, Data], output: Dict[str, Data]):

    json_data = open(input["input"].path, "r").read()

    prompt = """
    Construct a JSON-LD knowledge graph from the provided INPUT.
    Return only the JSON-LD in the following format:
    {{
        "@context": {{
            "...": "..."
        }},
        "...": "..."
    }}

    === INPUT ===

    {input}
    """.format(input=json_data)

    system_prompt = "You are a KG engineer. Generating JSON-LD KGs for a specific task."

    json_ld = get_client_from_env().send_prompt(prompt, "json", system_prompt=system_prompt)



@Registry.task(
    description="Construct a knowledge graph from any input with a knowledge graph.",
    input_spec={"input": DataFormat.ANY},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["Construction", "LLM"]
)
def construct_from_any_input_with_ontology_as_json_ld(input: Dict[str, Data], output: Dict[str, Data]):
    
    ontology_path = os.environ.get("ONTOLOGY_PATH")
    if not ontology_path:
        raise ValueError("ONTOLOGY_PATH is not set")
    
    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    ontology_snippet = generate_ontology_snippet_v2(ontology)

    prompt = """
Construct a knowledge graph from the provided INPUT and ONTOLOGY.
Generate at least a type and a label triple for each entity.
Return only JSON-LD in the following format:

{{
    "@context": {{
        "...": "..."
    }},
    "...": "..."
}}

=== ONTOLOGY ===

{ontology_snippet}

=== INPUT ===
""".format(ontology_snippet=ontology_snippet)

    system_prompt = "You are a KG engineer. Generating JSON-LD output for a specific task."
    llm = get_client_from_env()
    execute_prompt_for_input(input, output, prompt, system_prompt, llm, "json")

@Registry.task(
    description="Construct a knowledge graph from any input with a knowledge graph.",
    input_spec={"input": DataFormat.ANY, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["Construction", "LLM"]
)
def construct_from_any_input_with_kg_as_json_ld(input: Dict[str, Data], output: Dict[str, Data]):
    
    kg_path: Path = input["target"].path
    graph = __parse_rdf_file_to_graph(kg_path)
    kg_data = graph.serialize(format="turtle")

    prompt = """
Construct a knowledge graph from the provided INPUT,
for the given current TARGET knowledge graph RDF Turtle.
Only use the TARGET KG's exisiting properties and classes/types.
Try to reuse exisitng entity URIs, if there is a possible match, else new URI.
Return only the JSON-LD in the following format:
{{
    "@context": {{
        "...": "..."
    }},
    "...": "..."
}}

=== TARGET ===

{target}

=== INPUT ===
""".format(target=kg_data)

    system_prompt = "You are a KG engineer. Generating JSON-LD output for a specific task."
    llm = get_client_from_env()
    execute_prompt_for_input(input, output, prompt, system_prompt, llm, "json")


@Registry.task(
    description="Construct a knowledge graph from any input with a knowledge graph.",
    input_spec={"input": DataFormat.ANY, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["Construction", "LLM"]
)
def construct_from_any_input_with_kg_and_ontology_as_json_ld(input: Dict[str, Data], output: Dict[str, Data]):
    pass