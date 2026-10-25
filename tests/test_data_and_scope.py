import pandas as pd
import pytest

from co2_predictor.config import DATA_PATH, TARGET
from co2_predictor.data.loader import CO2DataLoader
from co2_predictor.features.selection import FeatureScope

FUEL_COLUMNS = (
    "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)",
    "Fuel Consumption Comb (L/100 km)",
    "Fuel Consumption Comb (mpg)",
)


def make_scope() -> FeatureScope:
    # Same scope as scripts/run_training.py
    return FeatureScope(
        redundant_fuel_columns=("Fuel Consumption Comb (mpg)",),
        high_cardinality_columns=("Model",),
        inference_unavailable_columns=FUEL_COLUMNS[:3],
    )


def test_loader_drops_duplicates(tmp_path, raw_df):
    path = tmp_path / "cars.csv"
    raw_df.to_csv(path, index=False)
    clean = CO2DataLoader(path).load_clean()
    assert len(clean) == len(raw_df) - 1
    assert not clean.duplicated().any()


def test_real_dataset_loads_with_target():
    df = CO2DataLoader(DATA_PATH).load_clean()
    assert TARGET in df.columns
    assert len(df) > 5000
    assert df[TARGET].notna().all()


def test_feature_scope_removes_post_production_columns(raw_df):
    """Guards the project's core claim: no fuel-consumption leakage."""
    out = make_scope().apply(raw_df)
    for col in FUEL_COLUMNS:
        assert col not in out.columns
    assert "Model" not in out.columns
    assert TARGET in out.columns


def test_feature_scope_keeps_pre_production_specs(raw_df):
    out = make_scope().apply(raw_df)
    expected = {"Make", "Vehicle Class", "Engine Size(L)", "Cylinders",
                "Transmission", "Fuel Type"}
    assert expected <= set(out.columns)


def test_feature_scope_ignores_missing_columns(raw_df):
    out = FeatureScope(high_cardinality_columns=("Does not exist",)).apply(raw_df)
    pd.testing.assert_frame_equal(out, raw_df)
