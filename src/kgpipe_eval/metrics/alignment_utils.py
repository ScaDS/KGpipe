from kgpipe.common import KG
from typing import Literal

CONFIG=None
# layz config dict
def get_config() -> dict:
    global CONFIG
    if CONFIG is None:
        # TODO
        pass
    return CONFIG

def get_aligned_entities(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[Entity]:
    return kg.entities.intersection(reference_kg.entities)

def get_aligned_triples(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[Triple]:
    return kg.triples.intersection(reference_kg.triples)

