import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from co2_predictor.config import TARGET
from co2_predictor.features.engineering import (
    add_cylinder_ratio,
    build_column_preprocessor,
)
from co2_predictor.features.selection import FeatureScope
from co2_predictor.inference.predictor import CO2Predictor, EXPECTED_COLUMNS
from co2_predictor.modeling.pipeline_factory import build_pipeline
from co2_predictor.persistence import load_pipeline, save_pipeline

NUMERICAL = ["Engine Size(L)", "Cylinders"]
CATEGORICAL = ["Make", "Vehicle Class", "Transmission", "Fuel Type"]


@pytest.fixture
def fitted_pipeline(raw_df):
    scope = FeatureScope(
        redundant_fuel_columns=("Fuel Consumption Comb (mpg)",),
        high_cardinality_columns=("Model",),
        inference_unavailable_columns=(
            "Fuel Consumption City (L/100 km)",
            "Fuel Consumption Hwy (L/100 km)",
            "Fuel Consumption Comb (L/100 km)",
        ),
    )
    df = scope.apply(raw_df)
    X, y = df.drop(columns=TARGET), df[TARGET]
    pre = build_column_preprocessor(numerical=NUMERICAL, categorical=CATEGORICAL)
    pipe = build_pipeline(preprocessor=pre, model=DecisionTreeRegressor(random_state=0))
    return pipe.fit(X, y), X


def test_cylinder_ratio_is_cylinders_over_engine_size():
    X = pd.DataFrame({"numeric__Cylinders": [4, 8], "numeric__Engine Size(L)": [2.0, 4.0]})
    out = add_cylinder_ratio(X)
    assert out["Cylinder per liter"].tolist() == [2.0, 2.0]


def test_cylinder_ratio_does_not_mutate_input():
    X = pd.DataFrame({"numeric__Cylinders": [4], "numeric__Engine Size(L)": [2.0]})
    add_cylinder_ratio(X)
    assert "Cylinder per liter" not in X.columns


def test_preprocessor_ignores_unseen_categories(fitted_pipeline):
    pipe, X = fitted_pipeline
    new = X.iloc[[0]].copy()
    new["Make"] = "NEVER_SEEN_BEFORE"
    pred = pipe.predict(new)
    assert pred.shape == (1,) and np.isfinite(pred).all()


def test_pipeline_predicts_one_value_per_row(fitted_pipeline):
    pipe, X = fitted_pipeline
    assert pipe.predict(X).shape == (len(X),)


def test_save_and_load_roundtrip(tmp_path, fitted_pipeline):
    pipe, X = fitted_pipeline
    path = tmp_path / "nested" / "model.joblib"  # parent dir is created on save
    save_pipeline(pipe, path)
    loaded = load_pipeline(path)
    np.testing.assert_allclose(pipe.predict(X), loaded.predict(X))


def test_load_missing_pipeline_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "missing.joblib")


def test_predictor_uses_only_pre_production_columns(tmp_path, fitted_pipeline):
    pipe, X = fitted_pipeline
    path = tmp_path / "model.joblib"
    save_pipeline(pipe, path)
    predictor = CO2Predictor(model_path=path)
    specs = X[list(EXPECTED_COLUMNS)].iloc[[0]]
    assert np.isfinite(predictor.predict(specs)).all()
    assert not any("Fuel Consumption" in c for c in EXPECTED_COLUMNS)
