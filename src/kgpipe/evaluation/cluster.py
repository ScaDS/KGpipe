from kgpipe.common.config import GLOBAL_STATE
from typing import Dict, List, Set, Optional  
import json
import csv
from rdflib import Graph, URIRef
from urllib.parse import urlparse
import hashlib

"""
matches CSV head
source1, source2, source3
entity1a, entity1b, entity1c
entity2a, entity2b, entity2c
entity3a, entity3b, entity3c
,entity4b, entity4c
"""


def hash_uri(uri: str) -> str:
    """
    Hash a URI using SHA-256.
    """
    return hashlib.sha256(uri.encode('utf-8')).hexdigest()

class MatchCluster:
    """
    A class to handle entity match clusters efficiently.
    
    This class stores clusters of matching entities and provides methods
    to query for matches within specific namespaces.
    """
    
    def __init__(self):
        # Dictionary mapping each URI to its cluster ID
        self.uri_to_cluster: Dict[str, int] = {}
        # Dictionary mapping cluster ID to set of URIs in that cluster
        self.clusters: Dict[int, Set[str]] = {}
        # Counter for generating unique cluster IDs
        self.next_cluster_id = 0
    
    def add_match(self, uri1: str, uri2: str):
        """
        Add a match between two URIs, merging their clusters if necessary.
        """
        cluster1 = self.uri_to_cluster.get(uri1)
        cluster2 = self.uri_to_cluster.get(uri2)
        
        if cluster1 is None and cluster2 is None:
            # Both URIs are new, create a new cluster
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            self.clusters[cluster_id] = {uri1, uri2}
            self.uri_to_cluster[uri1] = cluster_id
            self.uri_to_cluster[uri2] = cluster_id
            
        elif cluster1 is None:
            # uri1 is new, add it to uri2's cluster
            if cluster2 is not None:
                self.clusters[cluster2].add(uri1)
                self.uri_to_cluster[uri1] = cluster2
            
        elif cluster2 is None:
            # uri2 is new, add it to uri1's cluster
            if cluster1 is not None:
                self.clusters[cluster1].add(uri2)
                self.uri_to_cluster[uri2] = cluster1
            
        elif cluster1 != cluster2:
            # Both URIs are in different clusters, merge them
            if cluster1 is not None and cluster2 is not None:
                self._merge_clusters(cluster1, cluster2)
    
    def _merge_clusters(self, cluster1_id: int, cluster2_id: int):
        """
        Merge two clusters, keeping the smaller cluster ID.
        """
        if cluster1_id == cluster2_id:
            return
            
        # Determine which cluster to keep (the one with smaller ID)
        keep_id = min(cluster1_id, cluster2_id)
        remove_id = max(cluster1_id, cluster2_id)
        
        # Move all URIs from the cluster to be removed
        uris_to_move = self.clusters[remove_id]
        self.clusters[keep_id].update(uris_to_move)
        
        # Update URI mappings
        for uri in uris_to_move:
            self.uri_to_cluster[uri] = keep_id
            
        # Remove the old cluster
        del self.clusters[remove_id]
    
    def get_cluster(self, uri: str) -> Optional[Set[str]]:
        """
        Get the cluster containing the given URI.
        """
        cluster_id = self.uri_to_cluster.get(uri)
        if cluster_id is None:
            return None
        return self.clusters[cluster_id]
    
    def has_match_to_namespace(self, uri: str, target_ns: str) -> Optional[str]:
        """
        Check if the given URI has a match in the target namespace.
        Returns the matching URI if found, None otherwise.
        """
        cluster = self.get_cluster(uri)
        if cluster is None:
            return None
            
        for cluster_uri in cluster:
            if cluster_uri.startswith(target_ns):
                return cluster_uri
        return None

    def is_match(self, uri1: str, uri2: str, allow_match_on_suffix: bool = False) -> bool:
        """
        Check if two URIs are in the same match cluster.    
        """
        if allow_match_on_suffix:
            suffix1 = uri1.split("/")[-1]
            suffix2 = uri2.split("/")[-1]
            if suffix1 == suffix2:
                return True

        return self.get_cluster(uri1) == self.get_cluster(uri2)


    def __str__(self):
        return f"MatchCluster(uri_to_cluster={self.uri_to_cluster}, clusters={self.clusters})"

def load_matches(path: str, delimiter: str = '\t', gt_match_target_dataset: str = None) -> MatchCluster:
    """
    Load matches from matches CSV/TSV file.
    
    Args:
        path: Path to the matches file (CSV or TSV format)
        
    Returns:
        MatchCluster object containing all the match clusters
    """
    match_cluster = MatchCluster()
    
    with open(path, 'r', encoding='utf-8') as file:
        # Try to detect if it's TSV or CSV based on first line
        first_line = file.readline().strip()
        file.seek(0)  # Reset to beginning
        
        if delimiter in first_line:
            # TSV format (two columns: uri1, uri2)
            reader = csv.reader(file, delimiter=delimiter)
        else:
            # CSV format (multiple columns representing clusters)
            reader = csv.reader(file)
        
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
                
            # Filter out empty cells
            uris = [cell.strip() for cell in row if cell.strip()]
            
            if len(uris) == 2:
                # Simple pair format (uri1, uri2)
                match_cluster.add_match(uris[0], uris[1])
            elif len(uris) > 2:
                # Cluster format (uri1, uri2, uri3, ...)
                # Add matches between all pairs in the cluster
                for i in range(len(uris)):
                    for j in range(i + 1, len(uris)):
                        match_cluster.add_match(uris[i], uris[j])
    
    return match_cluster


def hasMatchToNs(uri: str, target_ns: str, match_cluster: Optional[MatchCluster] = None) -> str:
    """
    Checks match clusters for a given URI and their match to a URI in a given namespace.
    
    Args:
        uri: The URI to check for matches
        target_ns: The target namespace to look for matches in
        match_cluster: Optional MatchCluster instance. If None, uses global state.
        
    Returns:
        The matching URI in the target namespace, or empty string if no match found
    """
    if match_cluster is None:
        # Try to get from global state if available
        if 'match_cluster' in GLOBAL_STATE:
            match_cluster = GLOBAL_STATE['match_cluster']
        else:
            return ""
    
    if match_cluster is None:
        return ""
    
    result = match_cluster.has_match_to_namespace(uri, target_ns)
    return result if result is not None else ""

def is_match(uri1: str, uri2: str, match_cluster: Optional[MatchCluster] = None, allow_match_on_suffix: bool = False) -> bool:
    """
    Check if two URIs are in the same match cluster.
    """
    checker = False
    if uri2 == "http://kg.org/resource/b25598f9c0fce28a7700869fcb55d706":
        checker = True

    if allow_match_on_suffix:
        suffix1 = uri1.split("/")[-1]
        suffix2 = uri2.split("/")[-1]
        if suffix1 == suffix2:
            return True

    if match_cluster is None:
        return False
    if checker:
        print("uri1", uri1)
        print("uri2", uri2)
        print("match_cluster.get_cluster(uri1)", match_cluster.get_cluster(uri1))
        print("match_cluster.get_cluster(uri2)", match_cluster.get_cluster(uri2))

    c1 = match_cluster.get_cluster(uri1)
    c2 = match_cluster.get_cluster(uri2)
    if c1 is not None and c2 is not None:
        return c1 == c2
    else:
        return False


def load_matches_filtered(path: str, seed_URIs: Set[str], delimiter: str = '\t') -> MatchCluster:
    """
    Load matches from file but only include clusters that contain at least one seed URI.
    
    Args:
        path: Path to the matches file
        seed_URIs: Set of URIs to filter by
        
    Returns:
        MatchCluster containing only relevant clusters
    """
    full_cluster = load_matches(path, delimiter)
    filtered_cluster = MatchCluster()
    
    # Find all clusters that contain at least one seed URI
    relevant_clusters = set()
    for seed_uri in seed_URIs:
        cluster = full_cluster.get_cluster(seed_uri)
        if cluster:
            relevant_clusters.add(id(cluster))
    
    # Rebuild the filtered cluster with only relevant URIs
    for cluster_id, uris in full_cluster.clusters.items():
        if any(uri in seed_URIs for uri in uris):
            # Add all URIs from this cluster to the filtered cluster
            uri_list = list(uris)
            for i in range(len(uri_list)):
                for j in range(i + 1, len(uri_list)):
                    filtered_cluster.add_match(uri_list[i], uri_list[j])
    
    return filtered_cluster



def load_URIs_from_RDF(path: str) -> Set[str]:
    """
    Load URIs from an RDF file.
    """
    graph = Graph()
    graph.parse(path)
    s = set(str(s) for s in graph.subjects())
    o = set(str(o) for o in graph.objects() if isinstance(o, URIRef))
    return s.union(o)


def generate_match_clusters_file(uri_list_path: str, delimiter: str = '\t') -> MatchCluster:
    """
    Generate a match clusters file from a URI.
    """
    uri_list = load_URIs_from_RDF(uri_list_path)
    match_cluster = MatchCluster()
    for uri in uri_list:
        hash = hash_uri(uri)
        seed_uri = f"{hash}"
        rdf_uri = f"{hash}"
        text_uri = f"{hash}"

        # add match to cluster
        match_cluster.add_match(seed_uri, uri)
    return match_cluster


