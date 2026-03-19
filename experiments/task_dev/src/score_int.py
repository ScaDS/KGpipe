import math

from kgpipe_llm.common.core import get_client_from_env

def twcs():
    pass

def wilson_interval(x, n, confidence=0.95):
    # z for common confidence levels; use scipy if you want arbitrary levels
    z_table = {0.90: 1.6448536269514722,
               0.95: 1.959963984540054,
               0.99: 2.5758293035489004}
    if confidence not in z_table:
        raise ValueError("Use confidence in {0.90, 0.95, 0.99} or swap in scipy.stats.norm.ppf.")
    z = z_table[confidence]

    phat = x / n
    denom = 1 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z / denom) * math.sqrt((phat*(1-phat)/n) + (z**2)/(4*n**2))
    return (center - half, center + half)

# Example usage:
# low, high = wilson_interval(x=420, n=750, confidence=0.95)
# print(low, high)

from rdflib import Graph, RDF, RDFS, Literal, URIRef, BNode

def graph_to_natural_language_statements(g: Graph) -> str:
    """
    Convert an RDF graph to a table of statements.

    == Input ==
    <p1> a <c1> .
    <p1> <label> "Alice" .
    <p1> <age> 30 .
    <p1> <knows> <p2>
    <p2> a <c1> .
    <p2> <label> "Bob" .
    <c1> <label> "Person" .

    == Output ==
    Alice<TAB>age<TAB>30
    Alice<TAB>is a<TAB>Person
    Alice<TAB>knows<TAB>Bob
    """
    def best_label(node) -> str:
        """Prefer rdfs:label; otherwise fall back to QName/localname/str."""
        if isinstance(node, Literal):
            return str(node)

        # rdfs:label lookup for URIRef/BNode
        lbls = list(g.objects(node, RDFS.label))
        if lbls:
            # Prefer any language-less label, else first
            for l in lbls:
                if isinstance(l, Literal) and (l.language is None):
                    return str(l)
            return str(lbls[0])

        if isinstance(node, URIRef):
            # Try QName from graph's namespace manager
            try:
                qn = g.namespace_manager.qname(node)
                # Use the local part of the QName if possible (e.g., ex:knows -> knows)
                if ":" in qn:
                    return qn.split(":", 1)[1]
                return qn
            except Exception:
                pass

            # Fallback: last fragment after # or /
            s = str(node)
            if "#" in s:
                return s.rsplit("#", 1)[1]
            if "/" in s:
                return s.rsplit("/", 1)[1]
            return s

        if isinstance(node, BNode):
            return f"_:{str(node)}"

        return str(node)

    def predicate_text(p) -> str:
        if p == RDF.type:
            return "is a"
        # Prefer rdfs:label for predicates too; otherwise localname/qname
        return best_label(p)

    statements = []
    seen = set()

    # Iterate triples in a stable(ish) order for reproducible output
    for s, p, o in sorted(g, key=lambda t: (str(t[0]), str(t[1]), str(t[2]))):
        # Skip label triples themselves (they’re used for rendering)
        if p == RDFS.label:
            continue

        subj = best_label(s)
        pred = predicate_text(p)
        obj = best_label(o)

        row = (subj, pred, obj)
        if row in seen:
            continue
        seen.add(row)
        statements.append("\t".join(row))

    return "\n".join(statements)


# 