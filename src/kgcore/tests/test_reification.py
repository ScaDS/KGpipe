# kgcore/tests/test_reification.py
import tempfile
import os
from kgcore.api import KG

def test_reification_metadata():
    """Test that metadata is properly stored using RDF reification."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_reification.ttl")
        
        # Create graph with RDF backend
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Create entities and relations
        ada = kg.create_entity(["Person"], {"name": "Ada", "born": 1815})
        charles = kg.create_entity(["Person"], {"name": "Charles", "born": 1791})
        knows_rel = kg.create_relation("KNOWS", ada.id, charles.id, {"since": 1843})
        
        # Add metadata using reification
        kg.add_meta(ada.id, {"confidence": 0.95, "source": "historical"})
        kg.add_meta(knows_rel.id, {"confidence": 0.9, "verified": True})
        
        # Verify the RDF file contains reification triples
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check for reification structure
        assert "rdf:Statement" in content
        assert "rdf:subject" in content
        assert "rdf:predicate" in content
        assert "rdf:object" in content
        assert "confidence" in content
        assert "source" in content
        
        print("✓ Reification test passed")

if __name__ == "__main__":
    test_reification_metadata()
