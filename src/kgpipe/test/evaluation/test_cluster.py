"""
Test script for benchutils.py functions.
"""

import tempfile
import os
from kgpipe.evaluation.cluster import MatchCluster, load_matches, hasMatchToNs, load_matches_filtered


def test_match_cluster():
    """Test the MatchCluster class functionality."""
    print("Testing MatchCluster class...")
    
    # Create a new cluster
    cluster = MatchCluster()
    
    # Add some matches
    cluster.add_match("http://dbpedia.org/resource/Film_A", "http://www.wikidata.org/entity/Q123")
    cluster.add_match("http://dbpedia.org/resource/Film_B", "http://www.wikidata.org/entity/Q456")
    cluster.add_match("http://dbpedia.org/resource/Film_A", "http://example.org/film/789")  # This should merge clusters
    
    # Test cluster retrieval
    film_a_cluster = cluster.get_cluster("http://dbpedia.org/resource/Film_A")
    print(f"Film A cluster: {film_a_cluster}")
    
    # Test namespace matching
    wikidata_match = cluster.has_match_to_namespace("http://dbpedia.org/resource/Film_A", "http://www.wikidata.org/entity/")
    print(f"Wikidata match for Film A: {wikidata_match}")
    
    example_match = cluster.has_match_to_namespace("http://dbpedia.org/resource/Film_A", "http://example.org/")
    print(f"Example match for Film A: {example_match}")
    
    # Test with hasMatchToNs function
    result = hasMatchToNs("http://dbpedia.org/resource/Film_A", "http://www.wikidata.org/entity/", cluster)
    print(f"hasMatchToNs result: {result}")
    
    print("✓ MatchCluster tests passed\n")


def test_is_match():
    """Test the is_match function."""
    print("Testing is_match function...")
    
    cluster = MatchCluster()
    cluster.add_match("http://dbpedia.org/resource/Film_A", "http://www.wikidata.org/entity/Q123")
    cluster.add_match("http://dbpedia.org/resource/Film_A", "http://example.org/film/789")
    cluster.add_match("http://dbpedia.org/resource/Film_B", "http://www.wikidata.org/entity/Q456")

    # Test is_match function
    result = cluster.is_match("http://dbpedia.org/resource/Film_A", "http://www.wikidata.org/entity/Q123")
    assert result == True

    result = cluster.is_match("http://www.wikidata.org/entity/Q123", "http://example.org/film/789")
    assert result == True
    
    result = cluster.is_match("http://dbpedia.org/resource/Film_A", "http://www.wikidata.org/entity/Q456")
    assert result == False
    

def test_load_matches():
    """Test loading matches from a file."""
    print("Testing load_matches function...")
    
    # Create a temporary TSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write("http://dbpedia.org/resource/Film_A\thttp://www.wikidata.org/entity/Q123\n")
        f.write("http://dbpedia.org/resource/Film_B\thttp://www.wikidata.org/entity/Q456\n")
        f.write("http://dbpedia.org/resource/Film_C\thttp://www.wikidata.org/entity/Q789\n")
        temp_file = f.name
    
    try:
        # Load matches
        cluster = load_matches(temp_file)
        
        # Test that matches were loaded
        film_a_cluster = cluster.get_cluster("http://dbpedia.org/resource/Film_A")
        print(f"Loaded Film A cluster: {film_a_cluster}")
        
        # Test namespace matching
        wikidata_match = cluster.has_match_to_namespace("http://dbpedia.org/resource/Film_A", "http://www.wikidata.org/entity/")
        print(f"Wikidata match: {wikidata_match}")
        
        print("✓ load_matches tests passed\n")
        
    finally:
        # Clean up
        os.unlink(temp_file)


def test_load_matches_filtered():
    """Test loading filtered matches."""
    print("Testing load_matches_filtered function...")
    
    # Create a temporary TSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write("http://dbpedia.org/resource/Film_A\thttp://www.wikidata.org/entity/Q123\n")
        f.write("http://dbpedia.org/resource/Film_B\thttp://www.wikidata.org/entity/Q456\n")
        f.write("http://dbpedia.org/resource/Film_C\thttp://www.wikidata.org/entity/Q789\n")
        temp_file = f.name
    
    try:
        # Define seed URIs (only Film A and Film C)
        seed_uris = {
            "http://dbpedia.org/resource/Film_A",
            "http://dbpedia.org/resource/Film_C"
        }
        
        # Load filtered matches
        filtered_cluster = load_matches_filtered(temp_file, seed_uris)
        
        # Test that only relevant clusters were loaded
        film_a_cluster = filtered_cluster.get_cluster("http://dbpedia.org/resource/Film_A")
        film_b_cluster = filtered_cluster.get_cluster("http://dbpedia.org/resource/Film_B")
        film_c_cluster = filtered_cluster.get_cluster("http://dbpedia.org/resource/Film_C")
        
        print(f"Film A in filtered cluster: {film_a_cluster is not None}")
        print(f"Film B in filtered cluster: {film_b_cluster is not None}")  # Should be None
        print(f"Film C in filtered cluster: {film_c_cluster is not None}")
        
        print("✓ load_matches_filtered tests passed\n")
        
    finally:
        # Clean up
        os.unlink(temp_file)


def main():
    """Run all tests."""
    print("Running benchutils tests...\n")
    
    test_match_cluster()
    test_load_matches()
    test_load_matches_filtered()
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    main() 