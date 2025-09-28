from rdflib import Namespace, RDF, RDFS, OWL, Literal, BNode, Graph, URIRef, XSD
from rdflib.namespace import SH

def generate_shacl_shapes_from_owl(onto: Graph) -> Graph:
    sh_graph = Graph()
    sh_graph.bind("sh", SH)
    b = lambda: BNode()

    # Cardinality from owl:Restriction
    for restr in onto.subjects(RDF.type, OWL.Restriction):
        on_prop = onto.value(restr, OWL.onProperty)
        target_class = None
        for cls in onto.subjects(RDFS.subClassOf, restr):
            target_class = cls
            break
        if not target_class or not on_prop:
            continue

        shape = URIRef(f"{str(target_class)}Shape")
        sh_graph.add((shape, RDF.type, SH.NodeShape))
        sh_graph.add((shape, SH.targetClass, target_class))

        prop_bnode = b()
        sh_graph.add((shape, SH.property, prop_bnode))
        sh_graph.add((prop_bnode, SH.path, on_prop))

        for pred, sh_pred in [
            (OWL.maxCardinality, SH.maxCount),
            (OWL.minCardinality, SH.minCount),
            (OWL.cardinality, SH.count)
        ]:
            val = onto.value(restr, pred)
            if isinstance(val, Literal):
                sh_graph.add((prop_bnode, sh_pred, Literal(int(val))))

    # Domain/range shapes (generic)
    for prop in onto.subjects(RDF.type, RDF.Property):
        domain = onto.value(prop, RDFS.domain)
        range_ = onto.value(prop, RDFS.range)
        if not domain and not range_:
            continue

        shape = URIRef(f"{str(prop)}Shape")
        sh_graph.add((shape, RDF.type, SH.NodeShape))
        sh_graph.add((shape, SH.targetSubjectsOf, prop))

        prop_bnode = b()
        sh_graph.add((shape, SH.property, prop_bnode))
        sh_graph.add((prop_bnode, SH.path, prop))

        if range_:
            if (range_, RDF.type, RDFS.Datatype) in onto or str(range_).startswith(str(XSD)):
                sh_graph.add((prop_bnode, SH.datatype, range_))
            else:
                sh_graph.add((prop_bnode, SH["class"], range_))

    return sh_graph

if __name__ == "__main__":
    shapes = generate_shacl_shapes_from_owl(onto)
    shapes.serialize("auto_shapes.ttl", format="turtle")

    from pyshacl import validate

    def validate_with_generated_shapes(data_file, shapes_file):
        conforms, _, report = validate(
            data_graph_path=data_file,
            shacl_graph_path=shapes_file,
            inference='rdfs',
        )
        print(report)