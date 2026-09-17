"""
Corresponds to notebook Sections 1.4, 3.5.2, 3.6.1, 3.6.2.

This module encodes the *fixed, non-learned* feature-scope decisions
of the project: which columns are structurally excluded because of
the pre-production deployment scenario (Section 1.4), redundancy
(3.5.2), or high cardinality (3.6.1). These are NOT fitted
transformers -- they don't learn anything from data, they just
enforce a decision that was already made and justified in the
notebook. Keeping them here (rather than inline in a script) makes
the decision testable and gives it one authoritative place to live,
instead of being duplicated wherever the dataframe is touched.

Consider whether this class should also be a valid step inside the
sklearn Pipeline (i.e. inherit sklearn.base.BaseEstimator,
TransformerMixin, implement fit/transform) so it can sit next to
ColumnTransformer in modeling/pipeline_factory.py, rather than being
a manual pandas-only step applied before the Pipeline. Either design
is defensible -- pick one and be able to explain why in an interview.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FeatureScope:
    """The fixed set of columns excluded from the model, and why.

    Fill in the three lists below directly from Sections 3.5.2 and
    3.6.1/3.6.2 of the notebook. Keeping them as named, documented
    class attributes (rather than magic strings scattered across
    the codebase) is the point of this class.
    """

    redundant_fuel_columns: tuple[str, ...] = field(default_factory=tuple)

    high_cardinality_columns: tuple[str, ...] = field(default_factory=tuple)

    inference_unavailable_columns: tuple[str, ...] = field(default_factory=tuple)

    def all_dropped_columns(self) -> tuple[str, ...]:
        """Union of every column this project excludes from modeling."""

        self.redundant_fuel_columns = ('Fuel Consumption Comb (mpg)',)
        self.high_cardinality_columns = ('Model',)
        self.inference_unavailable_columns = (
            'Fuel Consumption City (L/100 km)',
            'Fuel Consumption Hwy (L/100 km)',
            'Fuel Consumption Comb (L/100 km)'
        )


        return (
            'Model',
            'Fuel Consumption Comb (mpg)',
            'Fuel Consumption City (L/100 km)',
            'Fuel Consumption Hwy (L/100 km)',
            'Fuel Consumption Comb (L/100 km)'
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop all_dropped_columns() from df, if present.

        Moved from: Sections 3.5.2, 3.6.1, 3.6.2 drop() calls.
        """
        
        try:
            cols_to_drop = self.all_dropped_columns()
            return df.drop(columns=list(cols_to_drop))
        except Exception as e:
            return e
