from pydantic import BaseModel, AnyUrl
from typing import Optional, List, Literal as LiteralType

# Text extraction models

class TextTriple(BaseModel):
    head: str
    relation: str
    tail: str

class TextTriples(BaseModel):
    triples: List[TextTriple]

# RDF models

class IRI(BaseModel):
    iri: AnyUrl  # validates proper IRI/URL

class RDFObject(BaseModel):
    kind: LiteralType["iri", "literal"]
    value: str
    datatype: Optional[AnyUrl] = None      # required when kind == "literal" and the value is typed (xsd:dateTime, xsd:integer, etc.)
    language: Optional[str] = None         # BCP 47 tag; only for language-tagged strings

class RDFTriple(BaseModel):
    subject: IRI
    predicate: IRI
    object: RDFObject

class RDFTriples(BaseModel):
    triples: List[RDFTriple]