"""
Corresponds to notebook Section 4 (Feature Engineering experiment)
and Section 5 (build_preprocessor / build_features definitions).

Two separate things live here, don't blur them:

1. `add_cylinder_ratio` -- the ONE engineered feature that the
   Section 4 experiment justified keeping (Cylinder_pre_liter). The
   competing feature (Engine * Cylinders product) was tested and
   rejected in Section 4 -- do not port it here, only the winner.

2. `build_column_preprocessor` -- the ColumnTransformer factory
   (one-hot for categorical, passthrough for numeric), moved from
   Section 5's `build_preprocessor`.

Design note on naming: the original notebook function was called
`build_features`, which is easy to confuse with `FeatureScope`
(features/selection.py) or with feature *importance* elsewhere.
Consider renaming to something like `add_engineered_features` in the
modular version for clarity -- small thing, but naming collisions
across modules are exactly what makes modular code harder to read
than a notebook if you're not careful.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def add_cylinder_ratio(X: pd.DataFrame, prefix: str = "numeric") -> pd.DataFrame:
    """Add the Cylinders / Engine Size(L) ratio column.

    Moved from: Section 5, `build_features`.

    Note the `prefix` argument exists because this function is meant
    to run *after* the ColumnTransformer step inside a Pipeline (see
    modeling/pipeline_factory.py), consuming its prefixed output
    column names (e.g. 'numeric__Cylinders'). This only works if
    sklearn's set_config(transform_output="pandas") is active --
    see the note in config.py.
    """
    try:
        X = X.copy()
        X['Cylinder per liter'] = X[f'{prefix}__Cylinders'] / X[f'{prefix}__Engine Size(L)']
        return X
    except Exception as e:
        return e


def build_column_preprocessor(
    *, numerical: list[str], categorical: list[str]
) -> ColumnTransformer:
    """Build the ColumnTransformer (OneHotEncoder + passthrough).

    Moved from: Section 5, `build_preprocessor`.
    """
    try:
        categorical_prefix = 'OHE'
        numerical_prefix = 'numeric'
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    categorical_prefix,
                    OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False
                    ),
                    categorical
                ),
                (
                    numerical_prefix,
                    'passthrough',
                    numerical
                ),
            ]
        )

        return preprocessor
    except Exception as e:
        return e
