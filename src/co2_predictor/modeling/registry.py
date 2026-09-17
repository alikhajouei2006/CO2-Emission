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

# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

# BASELINE_MODELS = {
#     "LR": ...,
#     "Ridge": ...,
#     ...
# }  # Section 6

# TUNED_MODEL_REGISTRY = {
#     "Random Forest": {"estimator": ..., "param_distributions": {...}},  # Section 8.1
#     "XGB": {...},                                                       # Section 8.2
#     "Extra Trees": {...},                                               # Section 8.3
#     "Decision Tree": {...},                                             # Section 8.4
# }
