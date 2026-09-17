"""
Corresponds to notebook Section 9 (`metrics` dict + `evaluate_model`)
and Section 12.1-12.2 (final test-set metrics).

This is the ONLY function that computes R2/MAE/RMSE for a fitted
model against data. Both modeling/selector.py (train-set comparison
across candidate models) and scripts/run_training.py's final
test-set evaluation (once, for the winning model only) call this
same function -- do not fork it into a train version and a test
version, the whole point of the Section 11 fix was that "evaluate
metrics" shouldn't care which split it's given; the CALLER decides
which split is appropriate and how many times to call it.
"""

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

METRICS = {
    "R2": r2_score,
    "MAE": mean_absolute_error,
    "RMSE": root_mean_squared_error,
}
# Section 9


def evaluate_model(*, model, X, y, metrics: dict = METRICS) -> dict:
    """Return {metric_name: rounded_value} for model.predict(X) vs y.

    Moved from: Section 9, `evaluate_model`.
    """
    raise NotImplementedError
