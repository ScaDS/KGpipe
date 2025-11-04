# kgcore/tests/test_rdf_file.py
import tempfile
import os
from pathlib import Path
from kgcore.api import KG

def test_rdf_file_persistence():
    """Test basic RDF file persistence functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_kg.ttl")
        
        # Create graph with file persistence
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Create entities and relations
        ada = kg.create_entity(["Person"], {"name": "Ada", "born": 1815})
        charles = kg.create_entity(["Person"], {"name": "Charles", "born": 1791})
        machine = kg.create_entity(["Machine"], {"name": "Analytical Engine"})
        
        knows_rel = kg.create_relation("KNOWS", ada.id, charles.id, {"since": 1843})
        invented_rel = kg.create_relation("INVENTED", charles.id, machine.id, {"year": 1837})
        
        # Verify file was created
        assert os.path.exists(file_path)
        
        # Verify entities can be found
        persons = kg.find_entities("Person")
        assert len(persons) == 2
        assert any(p.props["name"] == "Ada" for p in persons)
        assert any(p.props["name"] == "Charles" for p in persons)
        
        # Verify relations can be found
        knows_rels = kg.find_relations("KNOWS")
        assert len(knows_rels) == 1
        assert knows_rels[0].props["since"] == "1843"
        
        # Test loading from existing file
        kg2 = KG(backend="rdf_file", file_path=file_path, format="turtle")
        loaded_persons = kg2.find_entities("Person")
        assert len(loaded_persons) == 2
        
        print("✓ RDF file persistence test passed")

if __name__ == "__main__":
    test_rdf_file_persistence()
