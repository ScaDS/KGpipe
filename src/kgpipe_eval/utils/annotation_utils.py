from typing import Literal

# Labels 

def get_labeled_entities(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[Entity]:
    return kg.entities.intersection(reference_kg.entities)

def get_labeled_triples(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[Triple]:
    return kg.triples.intersection(reference_kg.triples)


def label_triples_with_llm(Triple):
    """
You are validating RDF triples.

Task 1:
For each triple, decide whether it is:
- plausible in isolation
- implausible in isolation
- unclear

Task 2:
Considering that all triples refer to the same subject node, decide whether the set is:
- coherent
- ambiguous
- conflated
- temporally inconsistent
- geographically inconsistent

Task 3:
Explain which triples are mutually incompatible and why.
{
  "triple_labels": [
    {
      "triple": ":Paris :locatedIn :France .",
      "label": "plausible_in_isolation"
    },
    {
      "triple": ":Paris :population \"2,100,000\" .",
      "label": "plausible_in_isolation"
    },
    {
      "triple": ":Paris :locatedIn :Texas .",
      "label": "plausible_in_isolation"
    }
  ],
  "entity_label": "conflated",
  "graph_label": "contextually_incompatible",
  "explanation": "The subject :Paris appears to merge Paris, France and Paris, Texas."
}
    """