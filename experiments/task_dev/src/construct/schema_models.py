# pydantic >= 2.0
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, RootModel


# ---------- Shared config ----------
class JsonLDBaseModel(BaseModel):
    """
    Base class that:
    - Allows arbitrary JSON-LD properties (extra='allow')
    - Lets you populate fields by alias (so '@id', '@type', etc. work)
    - Keeps things permissive to handle varied JSON-LD inputs
    """
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=False,
        arbitrary_types_allowed=True,
        protected_namespaces=(),  # allow keys like 'model_*' if needed
    )


# ---------- JSON-LD types ----------
ContextValue = Union[str, Dict[str, Any]]
ContextType = Union[ContextValue, List[ContextValue]]
Direction = Literal["ltr", "rtl"]


class ValueObject(JsonLDBaseModel):
    value: Any = Field(alias="@value")
    type: Optional[str] = Field(None, alias="@type")
    language: Optional[str] = Field(None, alias="@language")
    direction: Optional[Direction] = Field(None, alias="@direction")


class ListObject(JsonLDBaseModel):
    # Each item can itself be a JSON-LD object (node, value, list, set)
    list: List["JsonLDObject"] = Field(alias="@list")


class SetObject(JsonLDBaseModel):
    set: List["JsonLDObject"] = Field(alias="@set")


class NodeObject(JsonLDBaseModel):
    """
    A JSON-LD Node Object. Unknown terms/properties are accepted via `extra='allow'`.
    """
    context: Optional[ContextType] = Field(None, alias="@context")
    id: Optional[str] = Field(None, alias="@id")
    type: Optional[Union[str, List[str]]] = Field(None, alias="@type")

    # Common JSON-LD keywords on nodes
    # reverse: Optional[Dict[str, "JsonLDObject"]] = Field(None, alias="@reverse")
    # index: Optional[str] = Field(None, alias="@index")
    # graph: Optional[List["NodeObject"]] = Field(None, alias="@graph")
    # included: Optional[List["NodeObject"]] = Field(None, alias="@included")


# A union of all JSON-LD object varieties
JsonLDObject = Union[NodeObject, ValueObject, ListObject, SetObject]


# Optional: top-level document can be a single node or an array of nodes
class JsonLDDocument(RootModel[Union[NodeObject, List[NodeObject]]]):
    pass


# If you use Python < 3.11 without `from __future__ import annotations`, uncomment:
# NodeObject.model_rebuild()
# ListObject.model_rebuild()
# SetObject.model_rebuild()
# ValueObject.model_rebuild()


# ---------- Examples ----------
if __name__ == "__main__":
    # 1) A simple node object (schema.org Person)
    person = NodeObject.model_validate(
        {
            "@context": "https://schema.org",
            "@type": "Person",
            "@id": "https://example.com/#me",
            "name": "Ada Lovelace",
            "knowsAbout": ["Mathematics", "Computing"],
            "affiliation": {
                "@type": "Organization",
                "name": "Analytical Engine Club",
            },
        }
    )
    print(person.model_dump_json(by_alias=True, exclude_none=True, indent=2))

    # 2) A value object with a language tag
    greeting = ValueObject.model_validate({"@value": "Hola", "@language": "es"})
    print(greeting.model_dump_json(by_alias=True, exclude_none=True, indent=2))

    # 3) A graph document
    graph_doc = NodeObject.model_validate(
        {
            "@context": "https://schema.org",
            "@graph": [
                {"@id": "ex:1", "@type": "Person", "name": "Alice"},
                {"@id": "ex:2", "@type": "Person", "name": "Bob"},
            ],
        }
    )
    print(graph_doc.model_dump_json(by_alias=True, exclude_none=True, indent=2))


JSON_LD_SCHEMA_DICT = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "JsonLDNodeObject",
  "type": "object",
  "additionalProperties": True,

  "properties": {
    "@context": {
      "anyOf": [
        { "type": "string" },
        { "type": "object", "additionalProperties": True },
        {
          "type": "array",
          "items": {
            "anyOf": [
              { "type": "string" },
              { "type": "object", "additionalProperties": True }
            ]
          }
        }
      ]
    },

    "@id": { "type": "string" },

    "@type": {
      "anyOf": [
        { "type": "string" },
        { "type": "array", "items": { "type": "string" } }
      ]
    },

    "@reverse": {
      "type": "object",
      "additionalProperties": { "$ref": "#/$defs/JsonLDObject" }
    },

    "@index": { "type": "string" },

    "@graph": {
      "type": "array",
      "items": { "$ref": "#/$defs/NodeObject" }
    },

    "@included": {
      "type": "array",
      "items": { "$ref": "#/$defs/NodeObject" }
    }
  },

  "$defs": {
    "JsonLDObject": {
      "anyOf": [
        { "$ref": "#/$defs/NodeObject" },
        { "$ref": "#/$defs/ValueObject" },
        { "$ref": "#/$defs/ListObject" },
        { "$ref": "#/$defs/SetObject" }
      ]
    },

    "NodeObject": {
      "type": "object",
      "additionalProperties": True,
      "properties": {
        "@id": { "type": "string" },
        "@type": {
          "anyOf": [
            { "type": "string" },
            { "type": "array", "items": { "type": "string" } }
          ]
        },
        "@reverse": {
          "type": "object",
          "additionalProperties": { "$ref": "#/$defs/JsonLDObject" }
        },
        "@index": { "type": "string" },
        "@graph": {
          "type": "array",
          "items": { "$ref": "#/$defs/NodeObject" }
        },
        "@included": {
          "type": "array",
          "items": { "$ref": "#/$defs/NodeObject" }
        }
      }
    },

    "ValueObject": {
      "type": "object",
      "additionalProperties": True,
      "required": ["@value"],
      "properties": {
        "@value": {},
        "@type": { "type": "string" },
        "@language": { "type": "string" },
        "@direction": { "enum": ["ltr", "rtl"] }
      }
    },

    "ListObject": {
      "type": "object",
      "additionalProperties": True,
      "required": ["@list"],
      "properties": {
        "@list": {
          "type": "array",
          "items": { "$ref": "#/$defs/JsonLDObject" }
        }
      }
    },

    "SetObject": {
      "type": "object",
      "additionalProperties": True,
      "required": ["@set"],
      "properties": {
        "@set": {
          "type": "array",
          "items": { "$ref": "#/$defs/JsonLDObject" }
        }
      }
    }
  }
}
