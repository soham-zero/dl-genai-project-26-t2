import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score

def clean_text_for_tfidf(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def get_options_list(row):
    return [str(row[opt]) for opt in ['A', 'B', 'C', 'D', 'E']]

def calculate_map3(predictions, ground_truth):
    """
    predictions: list of strings like "A C B"
    ground_truth: list/series of single letter strings like "A"
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

def get_top1_preds(predictions):
    """
    Extracts the top 1 prediction from a MAP@3 formatted string array.
    """
    return [pred.strip().split()[0] for pred in predictions]

def calculate_accuracy(predictions, ground_truth):
    """
    Calculates standard accuracy using the top-1 prediction.
    """
    return accuracy_score(ground_truth, get_top1_preds(predictions))

def calculate_f1(predictions, ground_truth):
    """
    Calculates Macro F1 score using the top-1 prediction.
    """
    return f1_score(ground_truth, get_top1_preds(predictions), average='macro')
