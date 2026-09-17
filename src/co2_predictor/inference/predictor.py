"""
Not present in the original notebook -- this is the "so what was
this all for" module. Everything else in this package exists to
produce ONE fitted Pipeline object; this class is what a downstream
consumer (a script, an API endpoint, another notebook) actually
imports and uses.

This is also where the pre-production framing from Section 1.4 pays
off concretely: predict() below should accept exactly the six raw
specification fields (Make, Vehicle Class, Engine Size, Cylinders,
Transmission, Fuel Type) and nothing else -- if someone tries to
pass a Fuel Consumption value in here, that's a sign the caller
misunderstands what this model is for, and arguably this class
should validate its input columns and reject anything outside that
set, rather than silently ignoring extra columns.
"""

from pathlib import Path

import pandas as pd

from co2_predictor.persistence import load_pipeline

EXPECTED_COLUMNS = (
    "Make",
    "Vehicle Class",
    "Engine Size(L)",
    "Cylinders",
    "Transmission",
    "Fuel Type",
)
# Section 5 / 1.4 -- the pre-production feature set.


class CO2Predictor:
    """Loads a persisted fitted pipeline and serves predictions.

    Never imports data/loader.py (no dedup logic belongs here) or
    evaluation/diagnostics.py (no plotting belongs here). This class
    should be usable with zero dependency on the rest of the training
    code, aside from the shared Pipeline object it loads.
    """

    def __init__(self, model_path: str | Path):
        self.pipeline = load_pipeline(model_path)

    def predict(self, vehicle_specs: pd.DataFrame) -> "pd.Series | float":
        """Predict CO2 Emissions(g/km) for one or more vehicles.

        vehicle_specs must contain exactly EXPECTED_COLUMNS.
        Consider validating that here and raising a clear error
        (e.g. ValueError listing missing/unexpected columns) rather
        than letting sklearn raise an opaque error three layers deep
        inside the ColumnTransformer.
        """
        raise NotImplementedError
