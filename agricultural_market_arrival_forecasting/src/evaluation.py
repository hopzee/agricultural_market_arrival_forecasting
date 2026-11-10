"""Chronological evaluation utilities and regression metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chronological_split(series: pd.Series, test_months: int):
    if test_months <= 0 or test_months >= len(series):
        raise ValueError("Test period must be positive and shorter than the series.")
    return series.iloc[:-test_months], series.iloc[-test_months:]


def chronological_three_way_split(series: pd.Series, train_fraction=0.70, validation_fraction=0.15):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than one.")
    n = len(series)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    if train_end < 15 or validation_end <= train_end or n - validation_end < 2:
        raise ValueError("Not enough observations for chronological train, validation and test periods. A longer uninterrupted series is required.")
    return series.iloc[:train_end], series.iloc[train_end:validation_end], series.iloc[validation_end:]


def calculate_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float); predicted = np.asarray(predicted, dtype=float)
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Actual and predicted arrays must have equal non-zero lengths.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    return {"MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan}


def evaluate_candidates(models, train: pd.Series, validation: pd.Series, test: pd.Series, fit_predict_fn):
    """Evaluate each model on validation, refit on train+validation, then test."""
    rows, predictions = [], {}
    for name in models:
        try:
            val_pred = fit_predict_fn(name, train, len(validation))
            val_metrics = calculate_metrics(validation.to_numpy(), val_pred.to_numpy())
            refit_history = pd.concat([train, validation])
            test_pred = fit_predict_fn(name, refit_history, len(test))
            test_metrics = calculate_metrics(test.to_numpy(), test_pred.to_numpy())
            predictions[name] = pd.DataFrame({"month": test.index, "actual_arrivals_t": test.to_numpy(), "predicted_arrivals_t": test_pred.to_numpy()})
            rows.append({"Model": name, "Validation MAE": val_metrics["MAE"], "Validation RMSE": val_metrics["RMSE"],
                "Test MAE": test_metrics["MAE"], "Test RMSE": test_metrics["RMSE"], "Test R²": test_metrics["R2"], "Status": "Completed"})
        except Exception as exc:
            rows.append({"Model": name, "Validation MAE": np.nan, "Validation RMSE": np.nan,
                "Test MAE": np.nan, "Test RMSE": np.nan, "Test R²": np.nan, "Status": f"Unavailable/failed: {exc}"})
    return pd.DataFrame(rows), predictions
