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