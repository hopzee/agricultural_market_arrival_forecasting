"""Forecasting algorithms with optional dependency detection."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from src.features import make_supervised, make_next_step_features, LAGS

BASE_MODELS = ["Naive", "Seasonal Naive", "ARIMA"]


def available_models() -> list[str]:
    names = BASE_MODELS.copy()
    try:
        import xgboost  # noqa: F401
        names.append("XGBoost")
    except Exception:
        pass
    try:
        import lightgbm  # noqa: F401
        names.append("LightGBM")
    except Exception:
        pass
    try:
        import tensorflow  # noqa: F401
        names.extend(["LSTM", "GRU"])
    except Exception:
        pass
    return names


def _future_index(index: pd.DatetimeIndex, steps: int) -> pd.DatetimeIndex:
    if len(index) == 0 or not isinstance(index, pd.DatetimeIndex):
        raise ValueError("A non-empty monthly DatetimeIndex is required.")
    return pd.date_range(index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")


def _tree_model(name: str):
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            random_state=42, n_jobs=1)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=250, max_depth=-1, learning_rate=0.03,
            num_leaves=15, random_state=42, verbosity=-1, n_jobs=1)
    raise ValueError(f"Unknown tree model: {name}")


def _sequence_forecast(name: str, train: pd.Series, steps: int) -> np.ndarray:
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except Exception as exc:
        raise ValueError("LSTM/GRU requires TensorFlow. Install requirements-sequence.txt for a compatible Python environment.") from exc
    window = 12
    if len(train) < 48:
        raise ValueError(f"{name} requires at least 48 contiguous monthly observations in this implementation.")
    tf.keras.utils.set_random_seed(42)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.to_numpy().reshape(-1, 1)).astype("float32").ravel()
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i]); y.append(scaled[i])
    X = np.asarray(X, dtype="float32").reshape(-1, window, 1)
    y = np.asarray(y, dtype="float32")
    recurrent = tf.keras.layers.LSTM(32) if name == "LSTM" else tf.keras.layers.GRU(32)
    network = tf.keras.Sequential([tf.keras.layers.Input(shape=(window, 1)), recurrent,
        tf.keras.layers.Dropout(0.1), tf.keras.layers.Dense(1)])
    network.compile(optimizer="adam", loss="mse")
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    network.fit(X, y, epochs=100, batch_size=16, validation_split=0.15,
        shuffle=False, verbose=0, callbacks=callbacks)
    history = list(scaled); predictions = []
    for _ in range(steps):
        sample = np.asarray(history[-window:], dtype="float32").reshape(1, window, 1)
        pred = float(network.predict(sample, verbose=0)[0, 0])
        predictions.append(pred); history.append(pred)
    return scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()


def fit_predict(name: str, train: pd.Series, steps: int) -> pd.Series:
    """Fit on supplied history and recursively forecast the requested months."""
    history = train.astype(float).sort_index()
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Monthly observations must have a DatetimeIndex.")
    if history.isna().any():
        raise ValueError("Model input contains missing arrivals. Missing reports are not zero-filled or bridged.")
    if len(history) < 15:
        raise ValueError("At least 15 contiguous monthly observations are required for this model workflow.")
    if steps < 1:
        raise ValueError("Forecast horizon must be at least one month.")
    if name == "Naive":
        predictions = np.repeat(float(history.iloc[-1]), steps)
    elif name == "Seasonal Naive":
        if len(history) < 12:
            raise ValueError("Seasonal Naive requires at least 12 monthly observations.")
        predictions = np.resize(history.iloc[-12:].to_numpy(), steps)
    elif name == "ARIMA":
        order = (1, 1, 1) if len(history) >= 24 else (1, 0, 0)
        fitted = ARIMA(history, order=order).fit()
        predictions = fitted.forecast(steps=steps).to_numpy()
    elif name in ("XGBoost", "LightGBM"):
        supervised = make_supervised(history)
        if len(supervised) < 12:
            raise ValueError(f"{name} requires at least 12 usable training rows after lag construction.")
        model = _tree_model(name)
        model.fit(supervised.drop(columns="target"), supervised["target"])
        rolling = history.copy(); predictions = []
        for date in _future_index(history.index, steps):
            prediction = float(model.predict(make_next_step_features(rolling, date))[0])
            predictions.append(prediction); rolling.loc[date] = prediction
        predictions = np.asarray(predictions)
    elif name in ("LSTM", "GRU"):
        predictions = _sequence_forecast(name, history, steps)
    else:
        raise ValueError(f"Unsupported model: {name}")
    predictions = np.maximum(np.asarray(predictions, dtype=float), 0.0)
    return pd.Series(predictions, index=_future_index(history.index, steps), name="forecast_arrivals_t")
