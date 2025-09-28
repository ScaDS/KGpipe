#!/usr/bin/env python3
"""
Demonstration script for benchutils.py functions.
"""

import sys
import os

# Add the kgflex src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kgpipe_tasks.benchutils import MatchCluster, load_matches, hasMatchToNs, load_matches_filtered


def demo_match_cluster():
    """Demonstrate the MatchCluster class functionality."""
    print("=== MatchCluster Demo ===\n")
    
    # Create a new cluster
    cluster = MatchCluster()
    
    # Add some matches (simulating the TSV data format)
    print("Adding matches...")
    cluster.add_match("http://dbpedia.org/resource/Saturday_Night_Fever", "http://www.wikidata.org/entity/Q47654")
    cluster.add_match("http://dbpedia.org/resource/The_Host_(2006_film)", "http://www.wikidata.org/entity/Q488222")
    cluster.add_match("http://dbpedia.org/resource/The_White_Tiger_(2021_film)", "http://www.wikidata.org/entity/Q101098258")
    
    # Add another match that should merge with an existing cluster
    cluster.add_match("http://dbpedia.org/resource/Saturday_Night_Fever", "http://example.org/film/SNF1977")
    
    print(f"Total clusters: {len(cluster.clusters)}")
    print(f"Total URIs: {len(cluster.uri_to_cluster)}")
    
    # Test cluster retrieval
    snf_cluster = cluster.get_cluster("http://dbpedia.org/resource/Saturday_Night_Fever")
    print(f"\nSaturday Night Fever cluster: {snf_cluster}")
    
    # Test namespace matching
    wikidata_match = cluster.has_match_to_namespace("http://dbpedia.org/resource/Saturday_Night_Fever", "http://www.wikidata.org/entity/")
    print(f"Wikidata match for Saturday Night Fever: {wikidata_match}")
    
    example_match = cluster.has_match_to_namespace("http://dbpedia.org/resource/Saturday_Night_Fever", "http://example.org/")
    print(f"Example match for Saturday Night Fever: {example_match}")
    
    # Test with hasMatchToNs function
    result = hasMatchToNs("http://dbpedia.org/resource/Saturday_Night_Fever", "http://www.wikidata.org/entity/", cluster)
    print(f"hasMatchToNs result: {result}")
    
    print("\n✓ MatchCluster demo completed\n")


def demo_load_matches():
    """Demonstrate loading matches from a file."""
    print("=== Load Matches Demo ===\n")
    
    # Create a temporary TSV file with sample data
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write("http://dbpedia.org/resource/Saturday_Night_Fever\thttp://www.wikidata.org/entity/Q47654\n")
        f.write("http://dbpedia.org/resource/The_Host_(2006_film)\thttp://www.wikidata.org/entity/Q488222\n")
        f.write("http://dbpedia.org/resource/The_White_Tiger_(2021_film)\thttp://www.wikidata.org/entity/Q101098258\n")
        temp_file = f.name
    
    try:
        print(f"Loading matches from: {temp_file}")
        
        # Load matches
        cluster = load_matches(temp_file)
        
        print(f"Loaded {len(cluster.clusters)} clusters")
        print(f"Loaded {len(cluster.uri_to_cluster)} URIs")
        
        # Test that matches were loaded
        snf_cluster = cluster.get_cluster("http://dbpedia.org/resource/Saturday_Night_Fever")
        print(f"\nSaturday Night Fever cluster: {snf_cluster}")
        
        # Test namespace matching
        wikidata_match = cluster.has_match_to_namespace("http://dbpedia.org/resource/Saturday_Night_Fever", "http://www.wikidata.org/entity/")
        print(f"Wikidata match: {wikidata_match}")
        
        print("\n✓ Load matches demo completed\n")
        
    finally:
        # Clean up
        os.unlink(temp_file)


def demo_load_matches_filtered():
    """Demonstrate loading filtered matches."""
    print("=== Load Filtered Matches Demo ===\n")
    
    # Create a temporary TSV file
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write("http://dbpedia.org/resource/Saturday_Night_Fever\thttp://www.wikidata.org/entity/Q47654\n")
        f.write("http://dbpedia.org/resource/The_Host_(2006_film)\thttp://www.wikidata.org/entity/Q488222\n")
        f.write("http://dbpedia.org/resource/The_White_Tiger_(2021_film)\thttp://www.wikidata.org/entity/Q101098258\n")
        temp_file = f.name
    
    try:
        # Define seed URIs (only Saturday Night Fever and The White Tiger)
        seed_uris = {
            "http://dbpedia.org/resource/Saturday_Night_Fever",
            "http://dbpedia.org/resource/The_White_Tiger_(2021_film)"
        }
        
        print(f"Seed URIs: {seed_uris}")
        
        # Load filtered matches
        filtered_cluster = load_matches_filtered(temp_file, seed_uris)
        
        print(f"Filtered cluster contains {len(filtered_cluster.clusters)} clusters")
        print(f"Filtered cluster contains {len(filtered_cluster.uri_to_cluster)} URIs")
        
        # Test that only relevant clusters were loaded
        snf_cluster = filtered_cluster.get_cluster("http://dbpedia.org/resource/Saturday_Night_Fever")
        host_cluster = filtered_cluster.get_cluster("http://dbpedia.org/resource/The_Host_(2006_film)")
        wt_cluster = filtered_cluster.get_cluster("http://dbpedia.org/resource/The_White_Tiger_(2021_film)")
        
        print(f"\nSaturday Night Fever in filtered cluster: {snf_cluster is not None}")
        print(f"The Host in filtered cluster: {host_cluster is not None}")  # Should be None
        print(f"The White Tiger in filtered cluster: {wt_cluster is not None}")
        
        print("\n✓ Load filtered matches demo completed\n")
        
    finally:
        # Clean up
        os.unlink(temp_file)


def main():
    """Run all demonstrations."""
    print("=== KGbench Extras Benchutils Demo ===\n")
    
    demo_match_cluster()
    demo_load_matches()
    demo_load_matches_filtered()
    
    print("=== All demos completed successfully! ===")


if __name__ == "__main__":
    main() 