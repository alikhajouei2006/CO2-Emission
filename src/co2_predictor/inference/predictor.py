from pathlib import Path
import pandas as pd
from ..persistence import load_pipeline

EXPECTED_COLUMNS = (
    "Make",
    "Vehicle Class",
    "Engine Size(L)",
    "Cylinders",
    "Transmission",
    "Fuel Type",
)

class CO2Predictor:
    def __init__(self, model_path: str | Path):
        self.pipeline = load_pipeline(model_path)

    def predict(self, vehicle_specs: pd.DataFrame) -> "pd.Series | float":
        prediction = self.pipeline.predict(vehicle_specs)
        return prediction
