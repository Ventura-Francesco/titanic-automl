"""Threshold optimization for binary classification."""

import numpy as np
from sklearn.metrics import f1_score


def optimize_threshold(y_true, y_proba, metric='f1'):
    """Find optimal classification threshold using out-of-fold predictions.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy')
        
    Returns:
        float: Optimal threshold
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = np.mean(y_true == y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


def apply_threshold(y_proba, threshold=0.5):
    """Apply threshold to predicted probabilities.
    
    Args:
        y_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        np.ndarray: Binary predictions
    """
    return (y_proba >= threshold).astype(int)
