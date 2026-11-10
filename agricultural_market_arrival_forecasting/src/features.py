"""Causal lag and calendar feature generation for monthly arrival series."""
import pandas as pd

LAGS = (1, 2, 3, 6, 12)


def make_supervised(series: pd.Series) -> pd.DataFrame:
    if series.isna().any():
        raise ValueError("Lag features require a contiguous series without missing observations.")
    frame = pd.DataFrame({"target": series.astype(float)})
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["target"].shift(lag)
    frame["month"] = frame.index.month
    frame["year"] = frame.index.year
    return frame.dropna()


def make_next_step_features(history: pd.Series, date: pd.Timestamp) -> pd.DataFrame:
    row = {f"lag_{lag}": float(history.iloc[-lag]) for lag in LAGS}
    row.update({"month": int(date.month), "year": int(date.year)})
    return pd.DataFrame([row])
