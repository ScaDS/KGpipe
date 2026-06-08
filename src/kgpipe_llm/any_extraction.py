# Generalized variant of RDF triple generation

from kgpipe.common import Registry, DataFormat, Data, TaskInput, TaskOutput
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType, ConfigurationProfile
from kgpipe_llm.common.snippets import generate_ontology_snippet_v3
from kgcore.api.ontology import OntologyUtil
from pathlib import Path
from kgpipe_llm.common.core import LLMClient

from shutil import RegistryError
from pydantic import BaseModel, AnyUrl

# class OntologyGroundedSurfaceTriple(BaseModel):
#     subject_label: str
#     predicate_uri: AnyUrl
#     object_label: str

from pydantic import BaseModel, Field, AnyUrl


class SurfaceTriple(BaseModel):
    subject: str = Field(
        description="Surface-form subject label. Not a URI."
    )
    predicate_uri: AnyUrl = Field(
        description="Ontology property URI."
    )
    object: str = Field(
        description="Surface-form object label or literal. Not a URI."
    )


class SurfaceTripleExtractionResult(BaseModel):
    triples: list[SurfaceTriple]

# ontology-guided semantic triple extraction.
# surface semantic triples
# ontology-grounded surface triples


def get_ontology_grounded_surface_triples_prompt_template(ontology: str, input_data: str) -> str:
    return """
You are an ontology-guided semantic triple extraction system.

Your task is to extract ontology-grounded surface triples from the provided input data.

A valid triple has the form:

<subject, predicate_uri, object>

Where:
- subject is a surface-form string, label, name, or textual identifier.
- predicate_uri is a URI from the provided ontology vocabulary.
- object is a surface-form string, label, value, literal, or textual identifier.
- subject and object MUST NOT be converted into URIs.
- predicate_uri MUST be selected only from the ontology vocabulary.
- Do not invent ontology properties.
- Do not invent facts not supported by the input.
- Prefer the most specific ontology property that correctly matches the input.
- If a relation or attribute is present in the input but cannot be mapped to the ontology, place it in unmapped_candidates.
- Preserve meaningful entity names as they appear in the input, normalizing only whitespace and obvious formatting artifacts.
- Extract both attributes and relations when they can be represented with an ontology property.
- Return only valid structured output matching the provided schema.

Ontology vocabulary:

{ontology}

Input data:

{input_data}

Extraction guidance:
1. Identify named entities, records, rows, objects, or document subjects.
2. Identify attributes and relations expressed in the input.
3. Map each attribute or relation to the best matching ontology property URI.
4. Emit triples using string labels for subject and object.
5. Include evidence when possible.
6. Include confidence between 0.0 and 1.0.
7. Report unmapped relation or attribute candidates.
""".format(ontology=ontology, input_data=input_data)

def extract_ontology_surface_triples(data: str, ontology: Path, client: LLMClient) -> SurfaceTripleExtractionResult:

    ontology_snippet = generate_ontology_snippet_v3(OntologyUtil.load_ontology_from_file(ontology))

    prompt = get_ontology_grounded_surface_triples_prompt_template(ontology_snippet, data)
    response = client.send_prompt(prompt, SurfaceTripleExtractionResult)

    return response

@Registry.task(
    input_spec={"input": DataFormat.ANY},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Generate RDF triples for a schema",
    config_spec=ConfigurationDefinition(
        name="extract_ontology_surface_triples",
        parameters=[
            Parameter(
                name="ontology", 
                datatype=ParameterType.string, 
                description="The schema to generate RDF triples for"
            ),
            Parameter(
                name="prompt_template",
                datatype=ParameterType.string,
                description="The prompt template to use for the LLM"
            ),
        ]
    )
)
def extract_ontology_surface_triples_task(input: TaskInput, output: TaskOutput, config: ConfigurationProfile):
    pass



# from typing import Any, Literal
# from pydantic import BaseModel, Field, AnyUrl


# class OntologyTerm(BaseModel):
#     uri: AnyUrl = Field(
#         description="The ontology URI identifying a class, attribute, or relation."
#     )
#     label: str | None = Field(
#         default=None,
#         description="Optional human-readable label for the ontology term."
#     )
#     description: str | None = Field(
#         default=None,
#         description="Optional description or definition of the ontology term."
#     )


# class OntologyGroundedSurfaceTriple(BaseModel):
#     subject: str = Field(
#         description="Surface-form name or label of the subject entity. This is not a URI."
#     )

#     predicate_uri: AnyUrl = Field(
#         description="URI of the ontology property, attribute, or relation used as the predicate."
#     )

#     object: str = Field(
#         description="Surface-form value, entity name, label, literal, or textual object. This is not a URI."
#     )

#     subject_type_uri: AnyUrl | None = Field(
#         default=None,
#         description="Optional ontology class URI for the subject, if inferable from the input and ontology."
#     )

#     object_type_uri: AnyUrl | None = Field(
#         default=None,
#         description="Optional ontology class URI for the object, if inferable from the input and ontology."
#     )

#     evidence: str | None = Field(
#         default=None,
#         description="Short quote or compact excerpt from the input that supports this triple."
#     )

#     confidence: float = Field(
#         ge=0.0,
#         le=1.0,
#         description="Model confidence that the triple is correct and uses the appropriate ontology predicate."
#     )


# class TripleExtractionIssue(BaseModel):
#     message: str = Field(
#         description="Description of an ambiguity, missing ontology term, or extraction problem."
#     )

#     severity: Literal["info", "warning", "error"] = Field(
#         description="Severity of the issue."
#     )

#     related_text: str | None = Field(
#         default=None,
#         description="Optional source text related to the issue."
#     )


# class OntologySurfaceTripleExtractionResult(BaseModel):
#     triples: list[OntologyGroundedSurfaceTriple] = Field(
#         description="Extracted ontology-grounded surface triples."
#     )

#     unmapped_candidates: list[str] = Field(
#         default_factory=list,
#         description="Candidate relations or attributes found in the input that could not be mapped to the ontology."
#     )

#     issues: list[TripleExtractionIssue] = Field(
#         default_factory=list,
#         description="Warnings or errors encountered during extraction."
#     )