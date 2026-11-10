import numpy as np
import pandas as pd
import pytest
from src.evaluation import chronological_three_way_split, calculate_metrics


def test_chronological_split_preserves_order():
    idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    s = pd.Series(np.arange(60, dtype=float), index=idx)
    train, val, test = chronological_three_way_split(s)
    assert train.index.max() < val.index.min() < test.index.min()
    assert len(train) + len(val) + len(test) == 60


def test_metrics_are_correct_for_perfect_predictions():
    metrics = calculate_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["MAE"] == 0 and metrics["RMSE"] == 0 and metrics["R2"] == 1


def test_metrics_reject_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        calculate_metrics([1, np.nan], [1, 2])


def test_short_series_rejected_for_three_way_split():
    idx = pd.date_range("2024-01-01", periods=20, freq="MS")
    with pytest.raises(ValueError):
        chronological_three_way_split(pd.Series(range(20), index=idx))


def test_baseline_forecast_dates_and_nonnegative_values():
    from src.models import fit_predict
    idx = pd.date_range("2019-01-01", periods=36, freq="MS")
    series = pd.Series(np.linspace(10, 50, 36), index=idx)
    forecast = fit_predict("Naive", series, 4)
    expected = pd.date_range("2022-01-01", periods=4, freq="MS")
    assert forecast.index.equals(expected)
    assert (forecast >= 0).all()
