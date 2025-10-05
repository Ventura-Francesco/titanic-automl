"""Tests for threshold optimization module."""

import numpy as np
from titanic_automl.threshold import optimize_threshold, apply_threshold


def test_optimize_threshold_f1():
    """Test threshold optimization with F1 score."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.6, 0.4, 0.75, 0.85])
    
    threshold = optimize_threshold(y_true, y_proba, metric='f1')
    
    assert 0.1 <= threshold <= 0.9
    assert isinstance(threshold, float)


def test_optimize_threshold_accuracy():
    """Test threshold optimization with accuracy."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.2, 0.3, 0.7, 0.8, 0.6, 0.4, 0.75, 0.35])
    
    threshold = optimize_threshold(y_true, y_proba, metric='accuracy')
    
    assert 0.1 <= threshold <= 0.9
    assert isinstance(threshold, float)


def test_apply_threshold():
    """Test applying threshold to probabilities."""
    y_proba = np.array([0.2, 0.5, 0.7, 0.9, 0.3])
    
    y_pred = apply_threshold(y_proba, threshold=0.5)
    
    expected = np.array([0, 1, 1, 1, 0])
    assert np.array_equal(y_pred, expected)


def test_apply_threshold_custom():
    """Test applying custom threshold."""
    y_proba = np.array([0.2, 0.5, 0.7, 0.9, 0.3])
    
    y_pred = apply_threshold(y_proba, threshold=0.6)
    
    expected = np.array([0, 0, 1, 1, 0])
    assert np.array_equal(y_pred, expected)
