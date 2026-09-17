"""
Corresponds to notebook Section 5, `build_pipeline`.

Single factory function that assembles:
    ColumnTransformer -> add_cylinder_ratio -> StandardScaler -> [model]

This is the ONE function that both training (modeling/trainer.py)
and, indirectly, inference (via the persisted fitted object) depend
on. If you change the pipeline shape, this is the only place that
should need editing.
"""

from typing import Optional

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ..features.engineering import add_cylinder_ratio

def build_pipeline(
    *,
    preprocessor: ColumnTransformer,
    model: Optional[BaseEstimator],
) -> Pipeline:
    """Assemble the full sklearn Pipeline.

    Moved from: Section 5, `build_pipeline`.

    Keep the `model=None` case (a preprocessing-only pipeline) --
    Section 10.4's feature_names extraction in the notebook relies on
    being able to `.fit_transform()` a model-less version of this
    pipeline to recover output column names. That's an
    evaluation/diagnostics.py concern, but this factory needs to keep
    supporting it.
    """
    try:
        steps = [
            ('preprocess', preprocessor),
            (
                'feature_engineering',
                FunctionTransformer(
                    add_cylinder_ratio,
                    validate=False
                )
            ),
            ('scaler', StandardScaler()),
            ('model', model)
        ]

        pipeline = Pipeline(steps=steps)
        
        return pipeline
    except Exception as e:
        return e
