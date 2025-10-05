"""Utility functions for data loading and preprocessing."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Load train and test datasets.
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        
    Returns:
        tuple: (train_df, test_df)
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except FileNotFoundError:
        return None, None


def preprocess_data(df, is_train=True):
    """Preprocess the Titanic dataset.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data (has 'Survived' column)
        
    Returns:
        tuple: (X, y) if is_train else (X, None)
    """
    df = df.copy()
    
    # Extract target if training
    if is_train and 'Survived' in df.columns:
        y = df['Survived'].values
        df = df.drop('Survived', axis=1)
    else:
        y = None
    
    # Select features
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    available_cols = [col for col in feature_cols if col in df.columns]
    df = df[available_cols]
    
    # Encode categorical features
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def safe_div(a, b, default=0.0):
    """Safely divide two numbers, returning default if b is zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Value to return if b is zero
        
    Returns:
        float: a/b or default
    """
    return a / b if b != 0 else default
