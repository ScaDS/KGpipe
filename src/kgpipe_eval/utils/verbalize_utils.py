from kgpipe_eval.utils.kg_utils import Triple, TripleGraph, TriplePattern

def verbalize_triple_simple(triple: Triple, TripleGraph) -> str:
    """
    using label of subject, predicate, object to verbalize the triple
    """
    return f"{triple[0]} {triple[1]} {triple[2]}"

def verbalize_triples(triples: list[Triple]) -> list[str]:
    pass

def verbalize_triple_graph(triple_graph: TripleGraph) -> list[str]:
    pass

def verbalize_triple_graph_subject_groups(triple_graph: TripleGraph) -> list[list[str]]:
    pass