import numpy as np

def calculate_map3(predictions, ground_truth):
    """
    Calculates Mean Average Precision @ 3 (MAP@3).
    
    Args:
        predictions: List of strings, where each string contains space-separated predicted labels (e.g., 'A C B').
        ground_truth: List of true labels (e.g., 'A').
        
    Returns:
        float: The MAP@3 score.
    """
    scores = []
    for pred_str, gt in zip(predictions, ground_truth):
        pred_list = pred_str.strip().split()
        score = 0.0
        num_correct = 0
        for k, label in enumerate(pred_list[:3], start=1):
            if label == gt:
                num_correct += 1
                score += num_correct / k
        scores.append(score)
    return np.mean(scores)
