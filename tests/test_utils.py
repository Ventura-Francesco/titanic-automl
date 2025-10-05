"""Tests for utils module."""

import pytest
import numpy as np
import pandas as pd
from titanic_automl.utils import safe_div, preprocess_data


def test_safe_div_normal():
    """Test safe division with normal inputs."""
    assert safe_div(10, 2) == 5.0
    assert safe_div(7, 3) == pytest.approx(2.333, rel=0.01)


def test_safe_div_zero():
    """Test safe division by zero."""
    assert safe_div(10, 0) == 0.0
    assert safe_div(10, 0, default=999) == 999


def test_preprocess_data_with_target():
    """Test preprocessing with training data."""
    df = pd.DataFrame({
        'PassengerId': [1, 2, 3],
        'Survived': [0, 1, 0],
        'Pclass': [3, 1, 2],
        'Sex': ['male', 'female', 'male'],
        'Age': [22, 38, 26],
        'SibSp': [1, 0, 0],
        'Parch': [0, 1, 0],
        'Fare': [7.25, 71.28, 7.92],
        'Embarked': ['S', 'C', 'S']
    })
    
    X, y = preprocess_data(df, is_train=True)
    
    assert X.shape[0] == 3
    assert y is not None
    assert len(y) == 3
    assert np.array_equal(y, [0, 1, 0])


def test_preprocess_data_without_target():
    """Test preprocessing with test data (no target)."""
    df = pd.DataFrame({
        'PassengerId': [1, 2],
        'Pclass': [3, 1],
        'Sex': ['male', 'female'],
        'Age': [22, 38],
        'SibSp': [1, 0],
        'Parch': [0, 1],
        'Fare': [7.25, 71.28],
        'Embarked': ['S', 'C']
    })
    
    X, y = preprocess_data(df, is_train=False)
    
    assert X.shape[0] == 2
    assert y is None
