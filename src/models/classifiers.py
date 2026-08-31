"""
Defines the classifiers (Random Forest, XGBoost, Logistic Regression) and
their scikit-learn pipelines.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:  # fallback if xgboost isn't installed in the environment
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_XGBOOST = False

MODEL_NAMES = ["random_forest", "xgboost", "logistic_regression"]


def _make_estimator(model_name: str):
    if model_name == "random_forest":
        # Default reference config: 100 trees, unconstrained depth,
        # max_features = sqrt; exposed here as a GridSearchCV grid.
        return RandomForestClassifier(random_state=42), {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__max_features": ["sqrt"],
        }
    if model_name == "xgboost":
        if _HAS_XGBOOST:
            estimator = XGBClassifier(random_state=42, eval_metric="logloss")
        else:
            estimator = GradientBoostingClassifier(random_state=42)
        return estimator, {
            "clf__learning_rate": [0.01, 0.1, 0.3],
            "clf__max_depth": [4, 6, 8] if _HAS_XGBOOST else [3, 4, 5],
            "clf__n_estimators": [100],
        }
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=42), {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        }
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str) -> Pipeline:
    """Median imputation + standardization + classifier. Scaling is kept
    for every model for consistency, even for tree-based classifiers that
    wouldn't strictly need it."""
    estimator, _ = _make_estimator(model_name)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])


def build_param_grid(model_name: str) -> dict:
    _, grid = _make_estimator(model_name)
    return grid


def train_and_select(X, y, model_name: str, cv: int = 5, scoring: str = "f1_macro") -> GridSearchCV:
    """Grid search with stratified k-fold on the training set, optimizing
    macro F1-score."""
    pipeline = build_pipeline(model_name)
    grid = build_param_grid(model_name)
    min_class_count = int(y.value_counts().min()) if hasattr(y, "value_counts") else cv
    n_splits = max(2, min(cv, min_class_count))
    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, grid, cv=cv_strategy, scoring=scoring, n_jobs=-1)
    search.fit(X, y)
    return search
