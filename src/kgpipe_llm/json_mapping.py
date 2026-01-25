
from kgpipe_llm.common.core import get_client_from_env
from kgpipe_llm.common.snippets import generate_ontology_snippet_v3
from kgcore.api.ontology import Ontology, OntologyUtil
from pydantic import BaseModel
from typing import Optional, List, Literal as LiteralType, Callable, Dict
import json
from pathlib import Path
import os
from pydantic import AnyUrl
from rdflib import Graph, URIRef, Literal
from kgpipe.common import Registry, Data, DataFormat
from kgpipe_tasks.construction.json_sampler import exact_set_cover, load_json, enumerate_paths
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor



# def sample_json_files(folder: Path) -> list[str]:
#     files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")]
#     if not files:
#         print("No .json files found.")
#         return []

#     sets_by_doc = {}
#     for fp in files:
#         try:
#             data = load_json(fp)
#         except Exception as e:
#             print(f"Skipping {fp}: {e}")
#             continue
#         paths = enumerate_paths(
#             data,
#             include_containers=True,
#             wildcard_arrays=True,
#             leaf_only=False,
#             include_type_suffix=True,
#         )
#         sets_by_doc[fp] = paths

#     universe = set().union(*sets_by_doc.values()) if sets_by_doc else set()

#     picked, _  = exact_set_cover(universe, sets_by_doc)

#     return picked



# def llm_generate_rml_mapping(json_files: list[str], ontology: Graph) -> str:

#   ontology_ttl = ontology.serialize(format="turtle")

#   json_input = "\n".join([json.dumps(load_json(json_file)) for json_file in json_files])

#   prompt = f"""
# You are given:
# 1) A JSON snippet (the source data shape).
# 2) A Turtle (.ttl) ontology describing classes and properties to target.

# ## Goal
# Produce a **valid Turtle RML mapping** that maps the JSON to RDF conforming to the ontology. Use standard RML/R2RML prefixes and JSONPath. Assume the JSON is the logical source. Output **only** the Turtle mapping—no explanations.

# ## Strict requirements
# - Output must be valid Turtle syntax and valid RML (RML/R2RML vocab).
# - Use these prefixes (add more only if needed):
#   @prefix rml:   <http://semweb.mmlab.be/ns/rml#> .
#   @prefix rr:    <http://www.w3.org/ns/r2rml#> .
#   @prefix ql:    <http://semweb.mmlab.be/ns/ql#> .
#   @prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
#   @prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
#   @prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
# - Reuse ontology prefixes exactly as declared in the provided ontology (e.g., ex:).
# - For each rr:TriplesMap use rml:logicalSource with:
#   - rml:source "data.json" (use this literal as a placeholder),
#   - rml:referenceFormulation ql:JSONPath,
#   - rml:iterator "<ABSOLUTE JSONPath starting with $ selecting the repeating records>".
# - **JSONPath scoping rules (to avoid invalid paths):**
#   - Inside a TriplesMap, all `rml:reference`, `rr:child`, and `rr:parent` **MUST be RELATIVE to the TriplesMap’s `rml:iterator`** (i.e., **must NOT start with `$`**).
#   - Use `.` to refer to the **current node** when the iterator already points to the value (e.g., an array of strings).
#   - Use field names/relative paths like `name`, `starring[*]`, `producer[0].id` (no leading `$`).
#   - **Never end a JSONPath with `.` or `..`.**
# - Create one rr:TriplesMap per repeating “record” in the JSON that corresponds to a target class.
# - For each TriplesMap:
#   - Provide an rr:subjectMap with rr:template (prefer IRIs with a stable key).
#   - Include rr:class for the mapped ontology class.
#   - Add rr:predicateObjectMap entries for each property:
#     - Literals: use **relative** rml:reference "<JSONPath>" and, if inferable, rr:datatype (e.g., xsd:string, xsd:integer, xsd:boolean, xsd:dateTime).
#     - IRIs: use rr:termType rr:IRI with either rr:template (preferred) or **relative** rml:reference that yields CURIE/IRI strings.
# - Arrays:
#   - If a JSON field is an array of primitives (strings/numbers), point a single `rml:reference` to the **relative** array path (e.g., `tags[*]`) so each element yields a triple.
#   - If array items are objects that should become resources: create a separate TriplesMap with its own **absolute** iterator (e.g., `$.items[*]`) and relate them using:
#     - `rr:objectMap rr:parentTriplesMap + rr:joinCondition` where `rr:child` and `rr:parent` are **relative** paths, or
#     - `rr:template` that uses fields from the same record (if stable identifiers exist).
# - Nested objects:
#   - If nested objects represent distinct resources/classes, use a dedicated TriplesMap and link via parent/child join; keep join paths **relative**.
# - Null/absent fields: do not emit triples (RML engines will skip missing references).
# - Do not invent classes/properties—only use what exists in the ontology. If a perfect match is unclear, choose the closest semantically consistent properties.
# - Prefer stable IRI templates like: http://example.org/{id} or the ontology’s base if present.
# - Output only the Turtle mapping. No prose.

# ## Inputs
# ### JSON
# {json_input}

# ### Ontology (Turtle)
# {ontology_ttl}

# ## Deliverable
# A single Turtle document containing the RML mapping.
#   """

# #   prompt = f"""You are given:
# # 1) A JSON snippet (the source data shape).
# # 2) A Turtle (.ttl) ontology describing classes and properties to target.

# # ## Goal
# # Produce a **valid Turtle RML mapping** that maps the JSON to RDF conforming to the ontology. Use standard RML/R2RML prefixes and JSONPath. Assume the JSON is the logical source. Output **only** the Turtle mapping—no explanations.

# # ## Strict requirements
# # - Output must be valid Turtle syntax and valid RML (RML/R2RML vocab).
# # - Use these prefixes (add more only if needed):
# #   @prefix rml:   <http://semweb.mmlab.be/ns/rml#> .
# #   @prefix rr:    <http://www.w3.org/ns/r2rml#> .
# #   @prefix ql:    <http://semweb.mmlab.be/ns/ql#> .
# #   @prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
# #   @prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
# #   @prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
# # - Reuse ontology prefixes exactly as declared in the provided ontology (e.g., ex:).
# # - Use rml:logicalSource with:
# #   - rml:source "data.json" (use this literal as a placeholder),
# #   - rml:referenceFormulation ql:JSONPath,
# #   - rml:iterator "<JSONPath for the repeating records>".
# # - Create one rr:TriplesMap per repeating “record” in the JSON that corresponds to a target class.
# # - For each TriplesMap:
# #   - Provide an rr:subjectMap with rr:template (prefer IRIs with a stable key).
# #   - Include rr:class for the mapped ontology class.
# #   - Add rr:predicateObjectMap entries for each property:
# #     - Literals: use rml:reference "<JSONPath>" and, if inferable, rr:datatype (e.g., xsd:string, xsd:integer, xsd:boolean, xsd:dateTime).
# #     - IRIs: use rr:termType rr:IRI with either rr:template (preferred) or rml:reference that yields CURIE/IRI strings.
# # - Arrays: If a JSON field is an array, map it so that each element yields a separate triple:
# #   - If array items are primitive values (strings/nums): a single predicateObjectMap with rml:reference that points to the array path produces multiple triples.
# #   - If array items are objects that should become resources: create a separate TriplesMap for the item type with its own iterator, and relate them using either:
# #     - object map with rr:parentTriplesMap + rr:joinCondition, or
# #     - rr:template that uses fields from the same record (if stable identifiers exist).
# # - Nested objects:
# #   - If nested objects represent distinct resources/classes, use a dedicated TriplesMap and link via parent/child join.
# # - Null/absent fields: do not emit triples (RML engines will skip missing references).
# # - Do not invent classes/properties—only use what exists in the ontology. If a perfect match is unclear, choose the closest semantically consistent properties.
# # - Prefer stable IRI templates like: http://example.org/{id} or the ontology’s base if present.
# # - Output only the Turtle mapping. No prose.

# # ## Inputs
# # ### JSON
# # {json_input}

# # ### Ontology (Turtle)
# # {ontology_ttl}

# # ## Deliverable
# # A single Turtle document containing the RML mapping.
# #   """

#   system_prompt = """
#   You are a senior data-integration engineer who maps JSON into RDF using a given ontology.
#   You must return ONLY valid Turtle RML mapping that maps the JSON to RDF conforming to the ontology.
#   Do not include any explanations, comments, markdown, or extra keys.
#   If you cannot produce any mapping, return an empty string.
#   """

#   print(prompt)
#   # gpt-4o-mini
#   os.environ["DEFAULT_LLM_MODEL_NAME"] = "gpt-4.1"

#   response = get_client_from_env().send_prompt(prompt, "", system_prompt)

#   return str(response)

# def replace_rml_mapping_source(rml_mapping: str, json_file: str):
#   """
  
# <#PersonTriplesMap_Director>
#   rml:logicalSource [
#     rml:source "data.json" ;
#     rml:referenceFormulation ql:JSONPath ;
#     rml:iterator "$[*].director[*]"
#   ] ;
#   """
#   rml_graph = Graph()
#   rml_graph.parse(data=rml_mapping, format="turtle")

#   new_rml_graph = Graph()
#   for s, p, o in rml_graph:
#     if str(p) == "http://semweb.mmlab.be/ns/rml#source":
#       o = Literal(json_file)
#     new_rml_graph.add((s, p, o))

#   return new_rml_graph.serialize(format="turtle")

# def apply_rml_mapping(rml_mapping: str):
#   rml_graph = Graph()
#   rml_graph.parse(rml_mapping, format="turtle")
#   graph = Graph()
#   for s, p, o in rml_graph:
#     if str(p) == "http://semweb.mmlab.be/ns/rml#source":
#       o = Literal(json_file)
#     graph.add((s, p, o))
#   return graph.serialize(format="nt")
  

# def generate_rml_mapping_sampled():
#   folder = Path("/home/marvin/project/data/final/film_10k/split_0/sources/json/data/")
#   ontology_path = Path("/home/marvin/project/data/final/film_10k/ontology.ttl")
#   ontology = Graph()
#   ontology.parse(ontology_path, format="turtle")

#   picked = sample_json_files(folder)
#   possible_rml_mapping_ttl = llm_generate_rml_mapping(picked, ontology)
  
#   tmp_rml_mappings_dir = Path("/home/marvin/project/data/tmp/")
#   tmp_rml_mappings_dir.mkdir(parents=True, exist_ok=True)

#   for file in folder.iterdir():
#     if file.is_file() and file.suffix == ".json":
#       new_rml_mapping = replace_rml_mapping_source(possible_rml_mapping_ttl, file.as_posix())

#       with open(tmp_rml_mappings_dir / file.name.replace(".json", ".ttl"), "w") as f:
#         f.write(new_rml_mapping)

#   def call_rml_mapper(file: Path):
#     java = "/usr/share/java /home/marvin/.sdkman/candidates/java/17.0.15-tem/bin/java"
#     rmlmapper_jar = "/home/marvin/project/code/rmlmapper-8.0.0-r378-all.jar"

#     command = f"{java} -jar {rmlmapper_jar} -m {file.as_posix()} -s turtle -o {file.as_posix().replace(".ttl", ".nt")}"
#     print(command)

#     completed = subprocess.run(command, shell=True)
#     if completed.returncode != 0:
#       print(f"Error calling RML mapper: {completed.stderr}")
#       return False
#     return True

#   for file in list(tmp_rml_mappings_dir.iterdir())[:10]:
#     call_rml_mapper(file)

# if __name__ == "__main__":
#   generate_rml_mapping_sampled()

ONTOLOGY = None
def get_ontology() -> Ontology:
    global ONTOLOGY
    if ONTOLOGY is None:
        ontology_path = os.getenv("ONTOLOGY_PATH")
        if ontology_path is None:
            raise ValueError("ONTOLOGY is not set")
        ONTOLOGY = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    return ONTOLOGY

# --- Models ------------------------------------------------------------------

class IRI_v1(BaseModel):
    iri: AnyUrl  # validates proper IRI/URL

class RDF_Object_v1(BaseModel):
    kind: LiteralType["iri", "literal"]
    value: str
    datatype: Optional[AnyUrl] = None      # required when kind == "literal" and the value is typed (xsd:dateTime, xsd:integer, etc.)
    language: Optional[str] = None         # BCP 47 tag; only for language-tagged strings

class RDF_Triple_v1(BaseModel):
    subject: IRI_v1
    predicate: IRI_v1
    object: RDF_Object_v1

class RDF_Triples_v1(BaseModel):
    triples: List[RDF_Triple_v1]

# --- Utils ------------------------------------------------------------------

def run_parallel(func, file_pairs, parallel: int):
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(parallel) as p:
            p.starmap(func, file_pairs)
    except Exception:
        # Fallback to threads when pickling/importability is an issue
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            list(ex.map(lambda ab: func(*ab), file_pairs))


def apply_to_file_or_files_in_dir(func: Callable, input_path: Path, output_path: Path, parallel: int = 1) -> None:

    if os.path.isfile(input_path):
        func(input_path, output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        if parallel > 1:
            file_pairs = [
                (os.path.join(input_path, file), os.path.join(output_path, file.replace(".json", ".nt")))
                for file in os.listdir(input_path)
            ]

            run_parallel(func, file_pairs, parallel)
        else:
            for file in os.listdir(input_path):
                func(os.path.join(input_path, file), os.path.join(output_path, file))

# --- Mapper ------------------------------------------------------------------

class JSON_Mapping_v1:

    @staticmethod
    def construct_simple_rdf_triples(json_data: dict) -> RDF_Triples_v1:
        """
        Convert a JSON document into RDF triples guided by the provided ontology.
        """

        # Tight system prompt: role, constraints, and failure mode
        system_prompt = (
            "You are a senior data-integration engineer who maps JSON into RDF using a given ontology. "
            "You must return ONLY valid JSON that matches the exact schema provided. "
            "Do not include any explanations, comments, markdown, or extra keys. "
            "If you cannot produce any triples, return {\"triples\": []}."
        )

        # A robust, schema-aligned user prompt with clear rules + few-shot
        prompt = f"""
Return ONLY a JSON object of this exact shape (no markdown, no prose):

{{
  "triples": [
    {{
      "subject": {{"iri": "<subject-IRI>"}},
      "predicate": {{"iri": "<predicate-IRI>"}},
      "object": {{
        "kind": "iri" | "literal",
        "value": "<string>",
        "datatype": "<IRI-if-typed-or-null>",
        "language": "<bcp47-or-null>"
      }}
    }}
  ]
}}

=== Rules

1. Ontology & IRIs
- Use only ontology classes and properties.
- Subjects & predicates must be IRIs from the ontology.
- Objects: kind="iri" for resources, kind="literal" for values.

2. Literals
- Integers → xsd:integer, decimals → xsd:decimal, booleans → xsd:boolean.
- ISO-like dates/times → xsd:date, xsd:dateTime, or xsd:time.
- Plain strings → no datatype; add language if evident.
- If datatype is set, language must be null (and vice versa).

3. Identity & Subjects
- If JSON has a stable identifier (id, @id, uuid), mint subject IRIs accordingly; otherwise use ontology’s canonical pattern.
- Reuse the same subject IRI for all facts about that entity.

4. rdfs:label Triples
- For each distinct IRI subject or IRI object, emit at most one rdfs:label triple per language.
- Prefer ontology labels; otherwise use obvious JSON fields (name, title, etc.).
- Do not invent labels. Skip if none available.
- Label entities (subjects, IRI objects). Do not label predicates or external links unless a clear label exists.
- Keep language tags if present; otherwise omit.
- No duplicate labels.

5. Structure & Output
- Emit one triple per atomic fact; arrays → one triple per element.
- Ignore null/empty values.
- No duplicate triples.
- Return only the exact JSON structure defined (no comments, no extra keys).

=== Ontology

{generate_ontology_snippet_v3(get_ontology())}

=== Short Example (for format only)

Input:
  {{
    "id": "123",
    "name": "Alice",
    "homepage": "https://example.com/alice",
    "age": 30,
    "active": true
  }}

Output:
{{
  "triples": [
    {{
      "subject": {{ "iri": "https://example.org/person/123" }},
      "predicate": {{ "iri": "https://schema.org/name" }},
      "object": {{ "kind": "literal", "value": null, "lexical": "Alice", "datatype": null, "language": null }}
    }},
    {{
      "subject": {{ "iri": "https://example.org/person/123" }},
      "predicate": {{ "iri": "http://www.w3.org/2000/01/rdf-schema#label" }},
      "object": {{ "kind": "literal", "value": null, "lexical": "Person 123", "datatype": null, "language": "en" }}
    }},
    {{
      "subject": {{ "iri": "https://example.org/person/123" }},
      "predicate": {{ "iri": "https://schema.org/url" }},
      "object": {{ "kind": "iri", "value": "https://example.com/alice", "datatype": null, "language": null }}
    }},
    {{
      "subject": {{ "iri": "https://example.org/person/123" }},
      "predicate": {{ "iri": "https://schema.org/age" }},
      "object": {{ "kind": "literal", "value": "30", "datatype": "http://www.w3.org/2001/XMLSchema#integer", "language": null }}
    }}
  ]
}}

=== Input

{json.dumps(json_data, indent=4)}
"""

        # Ask the model for structured JSON matching RDF_Triples_v1
        response_dict = get_client_from_env().send_prompt(prompt, RDF_Triples_v1, system_prompt)

        return RDF_Triples_v1(**response_dict)

    @staticmethod
    def triples_to_graph(triples: RDF_Triples_v1) -> Graph:
        """
        Convert RDF_Triples_v1 into an rdflib.Graph.
        Handles IRIs, literals, datatypes, and language tags.
        """
        g = Graph()

        for t in triples.triples:
            subj = URIRef(str(t.subject.iri))
            pred = URIRef(str(t.predicate.iri))

            if t.object.kind == "iri":
                obj = URIRef(str(t.object.value))
            else:
                # Literal
                obj = Literal(
                    t.object.value or "",
                    datatype=URIRef(str(t.object.datatype)) if t.object.datatype else None,
                    lang=t.object.language if t.object.language else None
                )

            g.add((subj, pred, obj))

        return g

    @staticmethod
    def map_and_construct_json_file_to_rdf_file(input_path: Path, output_path: Path):
        json_data = json.load(open(input_path))
        triples = JSON_Mapping_v1.construct_simple_rdf_triples(json_data)
        graph = JSON_Mapping_v1.triples_to_graph(triples)
        graph.serialize(output_path, format="nt")

@Registry.task(
    description="Map and construct RDF files from JSON files",
    input_spec={"input": DataFormat.JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["RDF", "JSON Mapping", "LLM"]
)
def llm_task_map_and_construct(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    apply_to_file_or_files_in_dir(JSON_Mapping_v1.map_and_construct_json_file_to_rdf_file, input_path, output_path, parallel=8)


@Registry.task(
    description="Aggregate RDF files from directory",
    input_spec={"input": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["RDF", "JSON Mapping"]
)
def aggregate_rdf_files(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    final_graph = Graph()
    for file in input_path.iterdir():
        graph = Graph()
        try:
          graph.parse(file, format="nt")
          final_graph += graph
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    final_graph.serialize(outputs["output"].path, format="nt")