"""AutoML model search with hyperparameter optimization."""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score


def get_model_grid(demo_mode=True):
    """Get models and hyperparameter grids.
    
    Args:
        demo_mode: If True, use smaller search space for faster execution
        
    Returns:
        list: List of (name, model, param_grid) tuples
    """
    if demo_mode:
        models = [
            ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42),
             {'C': [0.1, 1.0]}),
            ('RandomForest', RandomForestClassifier(random_state=42),
             {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
        ]
    else:
        models = [
            ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42),
             {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l2']}),
            ('RandomForest', RandomForestClassifier(random_state=42),
             {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15, None],
              'min_samples_split': [2, 5, 10]}),
            ('GradientBoosting', GradientBoostingClassifier(random_state=42),
             {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2],
              'max_depth': [3, 5, 7]}),
            ('SVC', SVC(probability=True, random_state=42),
             {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}),
        ]
    
    return models


def search_best_model(X, y, demo_mode=True, cv=5):
    """Search for the best model using grid search.
    
    Args:
        X: Feature matrix
        y: Target vector
        demo_mode: If True, use smaller search space
        cv: Number of cross-validation folds
        
    Returns:
        tuple: (best_model_name, best_model, best_score, all_results)
    """
    models = get_model_grid(demo_mode=demo_mode)
    
    best_model = None
    best_model_name = None
    best_score = 0.0
    all_results = []
    
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, zero_division=0)
    
    for name, model, param_grid in models:
        print(f"Searching {name}...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_splitter, scoring=scorer,
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        score = grid_search.best_score_
        all_results.append({
            'model': name,
            'best_params': grid_search.best_params_,
            'best_score': score
        })
        
        print(f"  Best score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            best_model_name = name
    
    return best_model_name, best_model, best_score, all_results
