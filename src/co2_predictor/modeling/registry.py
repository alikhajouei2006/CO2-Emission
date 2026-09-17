"""
Corresponds to notebook Section 6 (baseline `models` dict) and
Section 8.1-8.4 (the four `param_dist_*` dictionaries).

This is the single place that lists "which models exist in this
project and what their search spaces are" -- so that
modeling/trainer.py can loop over it generically instead of having
four near-identical copy-pasted blocks (one per model), which is
what the notebook does in Section 8. That repetition was fine in a
notebook (explicit > clever, when you're exploring), but in a
package it's exactly the kind of thing that should be data, not
four duplicated code blocks -- if a fifth model is added later, you
want to add one registry entry, not copy-paste a whole new section.

Consider a small dataclass instead of a raw dict, e.g.:

    @dataclass(frozen=True)
    class ModelSpec:
        name: str
        estimator: BaseEstimator
        param_distributions: dict

and REGISTRY: list[ModelSpec] = [...]. Up to you -- a dict of dicts
also works fine, just be consistent.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ..config import RANDOM_STATE

BASELINE_MODELS = {
    "LR": LinearRegression(random_state=RANDOM_STATE),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "Elastic": ElasticNet(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesRegressor(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE)
}  

TUNED_MODEL_REGISTRY = {
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 20, 30, 50],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__bootstrap": [True, False]
        }
    },
    "XGB": {
        "estimator": XGBRegressor(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 5, 7, 10],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__gamma": [0, 0.1, 0.3, 0.5],
            "model__reg_alpha": [0, 0.01, 0.1, 1],
            "model__reg_lambda": [1, 5, 10]
        }
    },
    "Extra Trees": {
        "estimator": ExtraTreesRegressor(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 20, 30, 50],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None, 0.5, 0.8],
            "model__bootstrap": [False, True],
            "model__criterion": ["squared_error", "absolute_error"]
        }
    },
    "Decision Tree": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__max_depth": [None, 3, 5, 7, 10, 15, 20, 30],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [1.0, "sqrt", "log2"],
            "model__criterion": ["squared_error", "friedman_mse", "absolute_error"]
        }
    }
}