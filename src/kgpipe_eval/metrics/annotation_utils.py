

# Labels 

def get_labeled_entities(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[Entity]:
    return kg.entities.intersection(reference_kg.entities)

def get_labeled_triples(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[Triple]:
    return kg.triples.intersection(reference_kg.triples)