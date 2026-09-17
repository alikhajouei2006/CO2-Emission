from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class FeatureScope:
    redundant_fuel_columns: tuple[str, ...] = field(default_factory=tuple)

    high_cardinality_columns: tuple[str, ...] = field(default_factory=tuple)

    inference_unavailable_columns: tuple[str, ...] = field(default_factory=tuple)

    def all_dropped_columns(self) -> tuple[str, ...]:
        return (
            self.redundant_fuel_columns +
            self.high_cardinality_columns +
            self.inference_unavailable_columns
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = self.all_dropped_columns()
        return df.drop(columns=list(cols_to_drop), errors='ignore')
        
