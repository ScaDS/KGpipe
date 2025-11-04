# kgcore/tests/test_rdf_reification.py
from kgcore.backend.rdf_rdflib import RDFLibGraph

def test_rdf_reification():
    """Test reification in the base RDFLibGraph."""
    kg = RDFLibGraph()
    
    # Create entities and relations
    ada = kg.create_entity(["Person"], {"name": "Ada"})
    charles = kg.create_entity(["Person"], {"name": "Charles"})
    knows_rel = kg.create_relation("KNOWS", ada.id, charles.id)
    
    # Add metadata using reification
    kg.add_meta(ada.id, {"confidence": 0.95})
    kg.add_meta(knows_rel.id, {"confidence": 0.9})
    
    # Verify reification was added
    from rdflib.namespace import RDF
    statements = list(kg.g.triples((None, RDF.type, RDF.Statement)))
    assert len(statements) > 0, "No reification statements found"
    
    # Check that metadata properties are present
    confidence_triples = list(kg.g.triples((None, None, None)))
    confidence_found = any("confidence" in str(triple) for triple in confidence_triples)
    assert confidence_found, "Confidence metadata not found in reification"
    
    print("✓ RDF reification test passed")

if __name__ == "__main__":
    test_rdf_reification()
