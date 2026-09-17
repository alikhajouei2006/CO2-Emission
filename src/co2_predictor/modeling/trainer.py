"""
Corresponds to notebook Section 8 (Hyperparameter Optimization).

Generic replacement for the four repeated
RandomizedSearchCV(...).fit(X_train, y_train) blocks (8.1-8.4). Loops
over modeling.registry.TUNED_MODEL_REGISTRY instead of duplicating
the search code per model.

Design note (this is the resolution of the "40-minute retrain" issue
you flagged earlier, section 8): a notebook can get away with
`# best_rf = ...` commented-out manual reconstructions to avoid
re-running an expensive search. A package cannot. This is exactly
what persistence.py (joblib save/load of fitted searches or fitted
best_estimator_ objects) is for -- HyperparameterTuner.search() should
be a method you call ONCE per model and then persist the result, not
something re-run on every script invocation. Consider a `force=False`
flag or an explicit `--refit` CLI option in scripts/run_training.py
so a full re-tune is opt-in, not the default.
"""

from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold


class HyperparameterTuner:
    """Wraps RandomizedSearchCV with this project's fixed CV strategy.

    Moved from: Section 8 (the shared `cv = RepeatedKFold(...)`
    object, and the per-model `search_*` blocks).
    """

    def __init__(self, *, cv: RepeatedKFold, n_iter: int, random_state: int):
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state

    def search(self, *, pipeline, param_distributions: dict, X, y) -> RandomizedSearchCV:
        """Run RandomizedSearchCV for one model and return the fitted search.

        Moved from: Section 8.1-8.4, `search_rf.fit(X_train, y_train)`
        (and the XGB/ET/DT equivalents) -- but generic over any model,
        so it's called once per entry in TUNED_MODEL_REGISTRY instead
        of being copy-pasted four times.
        """
        raise NotImplementedError

    def search_all(self, *, registry: dict, preprocessor, X, y) -> dict:
        """Loop over the whole model registry, return {name: fitted_search}.

        This replaces Sections 8.1 through 8.4 in their entirety.
        """
        raise NotImplementedError
