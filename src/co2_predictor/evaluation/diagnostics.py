"""
Corresponds to notebook Section 10 (Model Diagnostics) and Section
12.3-12.5 (Final Residual/Parity/Permutation Importance).

*** ANALYSIS-ONLY MODULE ***
Nothing in this file is called by scripts/run_training.py's critical
path, and NOTHING here is ever imported by inference/predictor.py.
These functions exist so that a notebook (the original one, or any
future report) can call into tested, reusable plotting code instead
of redefining plot_residual/plot_parity/etc. inline. If you ever find
yourself importing this module from inference/ code, that's a sign
something has gone wrong architecturally -- diagnostics plots have no
business running when a live prediction is served.

Reminder from the fix you already made: every function here defaults
to being called with whatever split the CALLER passes in. Sections
10.x call these with (X_train, y_train) deliberately -- see the
Section 10 markdown note already in the notebook for why. Section
12.3-12.5 call the SAME functions with (X_test, y_test), exactly
once, for the final model only. Don't bake "train" or "test" into
the function names or defaults -- keep them split-agnostic, exactly
like evaluate_model in metrics.py.
"""

import matplotlib.figure
import numpy as np
import pandas as pd


def plot_learning_curve(model, X, y, model_name: str, *, cv: int) -> matplotlib.figure.Figure:
    """Moved from: Section 10.1, `plot_learning_curve`."""
    raise NotImplementedError


def plot_residual(*, model, X, y, model_name: str) -> matplotlib.figure.Figure:
    """Moved from: Section 10.2 / 12.3, `plot_residual`."""
    raise NotImplementedError


def plot_parity(*, model, X, y, model_name: str) -> matplotlib.figure.Figure:
    """Moved from: Section 10.3 / 12.4, `plot_parity` (fixed name --
    the notebook had this as `plot_praity` until the final review)."""
    raise NotImplementedError


def plot_feature_importance(
    *, model, feature_names, model_name: str
) -> matplotlib.figure.Figure:
    """Moved from: Section 10.4, `plot_feature_importance`.

    Note: relies on model.named_steps['model'].feature_importances_,
    which only exists for tree-based estimators -- this function is
    not generic over arbitrary sklearn models, document that.
    """
    raise NotImplementedError


def compute_permutation_importance(
    *, model, X, y, feature_names, n_repeats: int = 10
) -> pd.DataFrame:
    """Moved from: Section 10.5 / 12.5, `compute_permutation_importance`."""
    raise NotImplementedError
