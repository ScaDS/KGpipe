# TODO simple majority fusion class

def majority_fusion(values_with_scores):
    """
    Implements majority voting fusion strategy.
    
    Args:
        values_with_scores: List of tuples (value, score, source) where:
            - value: The actual value
            - score: Confidence score for this value
            - source: Source identifier (e.g., 'source_kg', 'target_kg')
    
    Returns:
        The value that appears most frequently, with ties broken by highest score
    """
    if not values_with_scores:
        return None
    
    # Count occurrences of each value
    value_counts = {}
    value_scores = {}
    
    for value, score, source in values_with_scores:
        if value not in value_counts:
            value_counts[value] = 0
            value_scores[value] = []
        
        value_counts[value] += 1
        value_scores[value].append(score)
    
    # Find the most frequent value(s)
    max_count = max(value_counts.values())
    most_frequent = [v for v, count in value_counts.items() if count == max_count]
    
    if len(most_frequent) == 1:
        return most_frequent[0]
    
    # Break ties by highest average score
    best_value = most_frequent[0]
    best_avg_score = sum(value_scores[best_value]) / len(value_scores[best_value])
    
    for value in most_frequent[1:]:
        avg_score = sum(value_scores[value]) / len(value_scores[value])
        if avg_score > best_avg_score:
            best_value = value
            best_avg_score = avg_score
    
    return best_value