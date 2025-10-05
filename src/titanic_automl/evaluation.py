"""Model evaluation utilities."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score


def evaluate_model(y_true, y_pred, y_proba=None):
    """Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        dict: Dictionary of metric scores
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    """Perform cross-validation on a model.
    
    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        dict: Cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }
