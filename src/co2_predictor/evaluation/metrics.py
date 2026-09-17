from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

METRICS = {
    "R2": r2_score,
    "MAE": mean_absolute_error,
    "RMSE": root_mean_squared_error,
}


def evaluate_model(*, model, X, y, metrics: dict = METRICS) -> dict:
    predicted_y = model.predict(X)

    results = {}
    for name, metric in metrics.items():
        results[name] = round(
            metric(y, predicted_y),
            2
        )

    return results