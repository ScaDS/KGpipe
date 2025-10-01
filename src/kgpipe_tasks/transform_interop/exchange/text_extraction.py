from typing import List, Optional
from pydantic import BaseModel, Field
from rdflib import Graph, URIRef

class TE_Span(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None
    surface_form: Optional[str] = None
    text: Optional[str] = None
    mapping: Optional[str] = None


class TE_Chains(BaseModel):
    main: Optional[str] = None
    aliases: List[TE_Span] = Field(default_factory=list)


class TE_Pair(BaseModel):
    """
    link_type: 'entity' or 'predicate' or 'type'
    """
    span: Optional[str] = None
    mapping: Optional[str] = None
    link_type: Optional[str] = None
    score: float

    # def __init__(self, mapping: str, span: str, link_type: str):
    #     self.span = span
    #     self.mapping = mapping
    #     if len(self.mapping) > 0 and self.mapping[0] == '<' and self.mapping[-1] == '>':
    #         self.mapping = self.mapping[1:-1]
    #     self.link_type = link_type

class TE_Triple(BaseModel):
    subject: TE_Span = Field(default_factory=TE_Span)
    predicate: TE_Span = Field(default_factory=TE_Span)
    object: TE_Span = Field(default_factory=TE_Span)

class TE_Document(BaseModel):
    text: Optional[str] = None
    triples: List[TE_Triple] = []
    chains: List[TE_Chains] = []
    links: List[TE_Pair] = []

    def set_text(self, text: str):
        self.text = text

    def add_triple(self, triple: TE_Triple):
        self.triples.append(triple)

    def add_chain(self, chain: TE_Chains):
        self.chains.append(chain)

    def add_link(self, link: TE_Pair):
        self.links.append(link)


def aggregate_te_documents(documents):
    result = TE_Document()
    for doc in documents:
        [ result.add_triple(triple) for triple in doc.triples ] 
        [ result.add_chain(chain) for chain in doc.chains ]
        [ result.add_link(link) for link in doc.links ]
    return result

def rdfFromtedoc(doc):
    g = Graph()

    for triple in doc.triples:
        g.add((URIRef(triple.subject.mapping), URIRef(triple.predicate.mapping), URIRef(triple.object.mapping)))

    return g