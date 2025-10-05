"""Command-line interface for Titanic AutoML."""

import argparse
import sys
import numpy as np

from .utils import load_data, preprocess_data
from .model_search import search_best_model
from .threshold import optimize_threshold, apply_threshold
from .evaluation import evaluate_model


def run_pipeline(demo_mode=True):
    """Run the full AutoML pipeline.
    
    Args:
        demo_mode: If True, use fast demo mode with smaller search space
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print(f"Titanic AutoML Pipeline ({'DEMO' if demo_mode else 'FULL'} mode)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data()
    
    if train_df is None:
        print("Warning: Could not load data files. Using synthetic data for testing.")
        # Generate synthetic data for testing
        np.random.seed(42)
        n_samples = 100 if demo_mode else 500
        n_features = 7
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    else:
        # Preprocess data
        print("Preprocessing data...")
        X, y = preprocess_data(train_df, is_train=True)
    
    print(f"Data shape: {X.shape}")
    print(f"Positive class ratio: {np.mean(y):.3f}")
    
    # Search for best model
    print("\nSearching for best model...")
    cv_folds = 3 if demo_mode else 5
    model_name, model, score, results = search_best_model(
        X, y, demo_mode=demo_mode, cv=cv_folds
    )
    
    print(f"\nBest model: {model_name}")
    print(f"CV F1 Score: {score:.4f}")
    
    # Train final model on all data
    print("\nTraining final model...")
    model.fit(X, y)
    
    # Optimize threshold
    print("Optimizing classification threshold...")
    y_proba = model.predict_proba(X)[:, 1]
    best_threshold = optimize_threshold(y, y_proba, metric='f1')
    print(f"Optimal threshold: {best_threshold:.3f}")
    
    # Evaluate
    y_pred = apply_threshold(y_proba, best_threshold)
    metrics = evaluate_model(y, y_pred, y_proba)
    
    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Titanic AutoML: Adaptive AutoML pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'full'],
        default='demo',
        help='Execution mode: demo (fast) or full (comprehensive)'
    )
    
    args = parser.parse_args()
    
    demo_mode = (args.mode == 'demo')
    
    try:
        exit_code = run_pipeline(demo_mode=demo_mode)
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
