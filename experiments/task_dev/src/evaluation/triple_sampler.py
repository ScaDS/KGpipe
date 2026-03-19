from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Any
import random

from rdflib import Graph

# Type alias for a triple (subject, predicate, object)
Triple = Tuple[Any, Any, Any]


def weighted_cluster_sampling(g: Graph, k: int) -> List[Triple]:
    """
    Weighted cluster sampling.
    With weighted clustering sampling (WCS), clusters are drawn
    with probabilities proportional to their sizes: πi = Mi /M, i = 1, ..., N.
    Then all triples in sampled clusters are evaluated. An unbiased es-
    timator of µ(G) is the Hansen-Hurwitz estimator

    A cluster is a set of triples that share a common subject.
    
    Args:
        g: RDFLib Graph containing the triples
        k: Number of clusters to sample
        
    Returns:
        List of all triples from the k sampled clusters
    """
    # Group triples by subject (clusters)
    clusters = defaultdict(list)
    for s, p, o in g.triples((None, None, None)):
        clusters[s].append((s, p, o))
    
    # Convert to list of (subject, triples) pairs
    cluster_list = list(clusters.items())
    
    if len(cluster_list) == 0:
        return []
    
    # Calculate total number of triples M
    M = sum(len(triples) for _, triples in cluster_list)
    
    if M == 0:
        return []
    
    # Calculate cluster sizes and weights
    cluster_sizes = [len(triples) for _, triples in cluster_list]
    weights = [size / M for size in cluster_sizes]
    
    # Sample k clusters with probabilities proportional to their sizes
    # Using sampling with replacement (appropriate for Hansen-Hurwitz estimator)
    num_clusters = len(cluster_list)
    sampled_indices = random.choices(
        range(num_clusters), 
        weights=weights, 
        k=k
    )
    
    # Collect all triples from sampled clusters
    result = []
    for idx in sampled_indices:
        _, triples = cluster_list[idx]
        result.extend(triples)
    
    return result



# Two-stage Weighted Cluster Sampling (TWCS)
# def twcs(g: Graph, k: int) -> List[Triple]:

import random

if __name__ == "__main__":
    clusters = {
        "cluster_1": [("s1", "p1", "o1"), ("s1", "p2", "o2")],
        "cluster_2": [("s2", "p1", "o1"), ("s2", "p2", "o2"), ("s2", "p3", "o3"), ("s2", "p4", "o4")],
        "cluster_3": [("s3", "p1", "o1"), ("s3", "p2", "o2"), ("s3", "p3", "o3")],
    }
    heads = list(clusters.keys())
    weights = [len(cluster) for cluster in clusters.values()]
    sampled_head = random.choices(heads, weights=weights, k=1)[0]

    # second-stage sampling
    pool = clusters[sampled_head]
    stageTwo = random.sample(pool, min(3, len(pool)))

    # get annotations for triples within sample
    sample = [[1 for triple in stageTwo]]
    print(sample)