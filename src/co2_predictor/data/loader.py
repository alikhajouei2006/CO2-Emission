from pathlib import Path
import pandas as pd


class CO2DataLoader:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        return df

    def load_clean(self) -> pd.DataFrame:
        df = self.load_raw()
        df = df.drop_duplicates()
        return df
