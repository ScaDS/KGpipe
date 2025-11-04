# kgcore/serialize/jsonld.py
from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel

def from_pydantic(model: BaseModel, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    doc = {"@type": model.__class__.__name__, **model.model_dump()}
    if context: doc["@context"] = context
    return doc

def from_entity(entity, base_iri: str = "http://example.org/"):
    return {
        "@id": f"{base_iri}{entity.id}",
        "@type": entity.labels,
        **entity.props
    }
