"""
Corresponds to notebook Section 2 (Data Loading & Initial Inspection).

Only the *actions* from Section 2 belong here (read_csv,
drop_duplicates). The *inspection* calls (df.shape, df.info(),
df.describe(), df.isna().sum(), the unique-values loop) were EDA --
they printed things for you to look at, they didn't transform data.
Do not port them here; they belong in the notebook only, or at most
in a throwaway analysis script. A production DataLoader should not
print anything by default.

Design note: duplicate removal (df.drop_duplicates()) is a
TRAINING-TIME-ONLY step. A single new vehicle spec arriving at
inference time is never "a duplicate" in a meaningful sense, and
CO2Predictor (inference/predictor.py) must NEVER call this class.
Keep that boundary explicit -- it's exactly the kind of distinction
a reviewer will check for.
"""

from pathlib import Path

import pandas as pd


class CO2DataLoader:
    """Loads and cleans the raw CO2 Emissions Canada dataset.

    Training-time only. See module docstring for why this must not
    be reused at inference time.
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def load_raw(self) -> pd.DataFrame:
        """Read the raw CSV from self.data_path.

        Moved from: Section 2, `df = pd.read_csv(DATA_PATH)`.
        """
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            return e

    def load_clean(self) -> pd.DataFrame:
        """load_raw() + drop_duplicates().

        Moved from: Section 2, `df = df.drop_duplicates()`.

        Consider logging (not printing) how many rows were dropped --
        useful for a training run log, not for stdout noise.
        """
        try:
            df = self.load_raw()
            df = df.drop_duplicates()
        except Exception as e:
            return e
