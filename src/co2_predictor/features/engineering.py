import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def add_cylinder_ratio(X: pd.DataFrame, prefix: str = "numeric") -> pd.DataFrame:
    X = X.copy()
    X['Cylinder per liter'] = X[f'{prefix}__Cylinders'] / X[f'{prefix}__Engine Size(L)']
    return X
   


def build_column_preprocessor(
    *, numerical: list[str], categorical: list[str]
) -> ColumnTransformer:
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