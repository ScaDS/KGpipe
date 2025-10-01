# pydantic >= 2.0
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, ValidationError, model_validator


# ---------- Shared ----------
Direction = Literal["ltr", "rtl"]


class JsonLDBaseModel(BaseModel):
    # Allow unknown keys, but we'll validate them in NodeObject's model_validator
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=False,
        protected_namespaces=(),
    )


# ---------- JSON-LD object variants ----------
class ValueObject(JsonLDBaseModel):
    value: Any = Field(alias="@value")
    type: Optional[str] = Field(None, alias="@type")
    language: Optional[str] = Field(None, alias="@language")
    direction: Optional[Direction] = Field(None, alias="@direction")


class ListObject(JsonLDBaseModel):
    # items must be JSON-LD objects (node/value/list/set)
    list: List["JsonLDObject"] = Field(alias="@list")


class SetObject(JsonLDBaseModel):
    # items must be JSON-LD objects (node/value/list/set)
    set: List["JsonLDObject"] = Field(alias="@set")


class NodeObject(JsonLDBaseModel):
    """
    JSON-LD Node Object with constrained additional properties:
    - Any non-@keyword property must be a JsonLDObject OR a list of JsonLDObject.
    - @reverse maps terms -> NodeObject OR list[NodeObject].
    """
    context: Optional[Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]] = Field(
        None, alias="@context"
    )
    id: Optional[str] = Field(None, alias="@id")
    type: Optional[Union[str, List[str]]] = Field(None, alias="@type")
    index: Optional[str] = Field(None, alias="@index")
    graph: Optional[List["NodeObject"]] = Field(None, alias="@graph")
    included: Optional[List["NodeObject"]] = Field(None, alias="@included")

    # @reverse must be node object(s), not value/list/set
    reverse: Optional[Dict[str, Union["NodeObject", List["NodeObject"]]]] = Field(
        None, alias="@reverse"
    )

    # ---- Enforce constraints on additional (non-@keyword) properties ----
    @model_validator(mode="after")
    def _validate_additional_properties(self) -> "NodeObject":
        # Fields known to the model (by alias—i.e., JSON-LD keywords)
        keyword_aliases = {
            f.alias for f in self.model_fields.values() if f.alias is not None
        }

        # Walk the instance dict; anything not a keyword alias is a "term" property
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if key in keyword_aliases:
                continue

            # Each term must be a JsonLDObject or list[JsonLDObject]
            try:
                self._ensure_jsonld_object_or_array(value)
            except ValidationError as e:
                raise e
            except Exception as e:
                # Wrap as a Pydantic validation error
                raise ValidationError.from_exception_data(
                    title="Invalid JSON-LD term value",
                    line_errors=[
                        {
                            "type": "value_error",
                            "loc": (key,),
                            "msg": f"Property '{key}' must be a JsonLDObject or an array of JsonLDObject. Got: {type(value).__name__}",
                            "input": value,
                        }
                    ],
                ) from e

        return self

    # Helper: validate a single JsonLDObject or a list of them
    @staticmethod
    def _ensure_jsonld_object_or_array(value: Any) -> None:
        def validate_single(v: Any) -> None:
            # Try NodeObject/ValueObject/ListObject/SetObject in turn
            for cls in (NodeObject, ValueObject, ListObject, SetObject):
                try:
                    cls.model_validate(v)
                    return  # ok
                except ValidationError:
                    continue
            raise TypeError("Not a JsonLDObject")

        if isinstance(value, list):
            for item in value:
                validate_single(item)
        else:
            validate_single(value)


# A union of all JSON-LD object varieties
JsonLDObject = Union[NodeObject, ValueObject, ListObject, SetObject]


# ---------- Optional: top-level document wrapper ----------
# If you want to enforce "document is a single NodeObject"
# you can just use NodeObject directly. If you want a list-of-nodes
# at top level, define your own wrapper model.
class JsonLDDocument(NodeObject):
    """Top-level JSON-LD document (a node)."""
    pass


# If you're on Python < 3.11 without `from __future__ import annotations`, call:
# NodeObject.model_rebuild()
# ListObject.model_rebuild()
# SetObject.model_rebuild()
# ValueObject.model_rebuild()

JSON_SCHEMA_DICT_v2 = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "JsonLDNodeObject",
  "type": "object",

  "properties": {
    # "@context": {
    #   "anyOf": [
    #     { "type": "string" },
    #     { "type": "object", "additionalProperties": true },
    #     {
    #       "type": "array",
    #       "items": {
    #         "anyOf": [
    #           { "type": "string" },
    #           { "type": "object", "additionalProperties": true }
    #         ]
    #       }
    #     }
    #   ]
    # },

    "@id": { "type": "string" },

    "@type": {
      "anyOf": [
        { "type": "string" },
        { "type": "array", "items": { "type": "string" } }
      ]
    },

    # "@index": { "type": "string" },

#     "@graph": {
#       "type": "array",
#       "items": { "$ref": "#/$defs/NodeObject" }
#     },

#     "@included": {
#       "type": "array",
#       "items": { "$ref": "#/$defs/NodeObject" }
#     },

    # "@reverse": {
    #   "type": "object",
    #   "additionalProperties": {
    #     "anyOf": [
    #       { "$ref": "#/$defs/NodeObject" },
    #       {
    #         "type": "array",
    #         "items": { "$ref": "#/$defs/NodeObject" }
    #       }
    #     ]
    #   }
    # }
  },

#  /* 🔒 Constrain non-@keyword properties on the root node: */
  "additionalProperties": {
    "anyOf": [
      { "$ref": "#/$defs/JsonLDObject" },
      {
        "type": "array",
        "items": { "$ref": "#/$defs/JsonLDObject" }
      }
    ]
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
      "properties": {
        "@id": { "type": "string" },
        "@type": {
          "anyOf": [
            { "type": "string" },
            { "type": "array", "items": { "type": "string" } }
          ]
        },
        "@index": { "type": "string" },
        "@graph": {
          "type": "array",
          "items": { "$ref": "#/$defs/NodeObject" }
        },
        "@included": {
          "type": "array",
          "items": { "$ref": "#/$defs/NodeObject" }
        },
        "@reverse": {
          "type": "object",
          "additionalProperties": {
            "anyOf": [
              { "$ref": "#/$defs/NodeObject" },
              {
                "type": "array",
                "items": { "$ref": "#/$defs/NodeObject" }
              }
            ]
          }
        }
      },
#      /* 🔒 Constrain non-@keyword properties on nested nodes too: */
      "additionalProperties": {
        "anyOf": [
          { "$ref": "#/$defs/JsonLDObject" },
          {
            "type": "array",
            "items": { "$ref": "#/$defs/JsonLDObject" }
          }
        ]
      }
    },

    "ValueObject": {
      "type": "object",
      "required": ["@value"],
      "properties": {
        "@value": {},
        "@type": { "type": "string" },
        "@language": { "type": "string" },
        "@direction": { "enum": ["ltr", "rtl"] }
      },
      "additionalProperties": False
    },

    "ListObject": {
      "type": "object",
      "required": ["@list"],
      "properties": {
        "@list": {
          "type": "array",
          "items": { "$ref": "#/$defs/JsonLDObject" }
        }
      },
      "additionalProperties": False
    },

    "SetObject": {
      "type": "object",
      "required": ["@set"],
      "properties": {
        "@set": {
          "type": "array",
          "items": { "$ref": "#/$defs/JsonLDObject" }
        }
      },
      "additionalProperties": False
    }
  }
}


import json

# ---------- Example usage ----------
if __name__ == "__main__":
    # Valid: term properties are JsonLDObject or array of JsonLDObject

    print(json.dumps(NodeObject.model_json_schema(), indent=4))

    exit()
    doc = JsonLDDocument.model_validate(
        {
            "@context": "https://schema.org",
            "@type": "Person",
            "name": {"@value": "Ada Lovelace"},                    # ValueObject
            "knowsAbout": [                                         # list of ValueObjects
                {"@value": "Mathematics", "@language": "en"},
                {"@value": "Computing"}
            ],
            "affiliation": {                                        # NodeObject
                "@type": "Organization",
                "name": {"@value": "Analytical Engine Club"}
            },
            "@reverse": {
                "alumniOf": [                                       # list of NodeObject
                    {"@type": "CollegeOrUniversity", "name": {"@value": "UCL"}}
                ]
            }
        }
    )

    # Serialize with '@' keys:
    print(doc.model_dump_json(by_alias=True, exclude_none=True, indent=2))
