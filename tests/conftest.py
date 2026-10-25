import pandas as pd
import pytest

from co2_predictor.config import initial_configurations


@pytest.fixture(scope="session", autouse=True)
def _configure_sklearn():
    """The pipeline relies on pandas output from ColumnTransformer."""
    initial_configurations()


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Tiny frame with the same columns as the real dataset (one duplicate row)."""
    rows = [
        ("ACURA", "ILX", "COMPACT", 2.0, 4, "AS5", "Z", 9.9, 6.7, 8.5, 33, 196),
        ("ACURA", "ILX", "COMPACT", 2.4, 4, "M6", "Z", 11.2, 7.7, 9.6, 29, 221),
        ("BMW", "X5", "SUV - SMALL", 3.0, 6, "AS8", "Z", 12.5, 9.0, 11.0, 26, 260),
        ("BMW", "X5", "SUV - SMALL", 4.4, 8, "AS8", "Z", 14.8, 10.5, 12.8, 22, 300),
        ("FORD", "F150", "PICKUP TRUCK", 5.0, 8, "A10", "X", 15.0, 11.0, 13.2, 21, 310),
        ("FORD", "F150", "PICKUP TRUCK", 2.7, 6, "A10", "X", 12.0, 9.5, 10.9, 26, 255),
        ("TOYOTA", "CAMRY", "MID-SIZE", 2.5, 4, "AS8", "X", 8.6, 5.8, 7.4, 38, 172),
        ("TOYOTA", "CAMRY", "MID-SIZE", 2.5, 4, "AS8", "X", 8.6, 5.8, 7.4, 38, 172),  # duplicate
    ]
    cols = [
        "Make", "Model", "Vehicle Class", "Engine Size(L)", "Cylinders",
        "Transmission", "Fuel Type",
        "Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)",
        "Fuel Consumption Comb (L/100 km)", "Fuel Consumption Comb (mpg)",
        "CO2 Emissions(g/km)",
    ]
    return pd.DataFrame(rows, columns=cols)
