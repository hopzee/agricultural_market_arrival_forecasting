"""Loading, validating and preparing monthly agricultural arrival data."""
from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {"month", "commodity", "state", "arrivals_t"}
KEY_COLUMNS = ["commodity", "state", "month"]


def load_dataset(file) -> pd.DataFrame:
    """Read CSV/CSV.GZ input and normalise column names."""
    name = str(getattr(file, "name", "")).lower()
    compression = "gzip" if name.endswith(".gz") else "infer"
    frame = pd.read_csv(file, compression=compression, low_memory=False)
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("The file contains duplicate column names after normalisation.")
    return frame


def validate_dataset(frame: pd.DataFrame) -> dict:
    errors, warnings = [], []
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        errors.append("Missing required columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        errors.append("The uploaded dataset contains no rows.")
    if not missing and not frame.empty:
        dates = pd.to_datetime(frame["month"], errors="coerce")
        if dates.isna().all():
            errors.append("The month column contains no dates that can be parsed.")
        elif dates.isna().any():
            warnings.append(f"{int(dates.isna().sum()):,} rows have invalid month values and will be excluded from modelling.")
        arrivals = pd.to_numeric(frame["arrivals_t"], errors="coerce")
        if arrivals.notna().sum() == 0:
            errors.append("arrivals_t contains no numeric values.")
        invalid_numeric = int(arrivals.isna().sum())
        if invalid_numeric:
            warnings.append(f"{invalid_numeric:,} arrival values are missing or non-numeric; they will not be treated as zero.")
        negative = int((arrivals < 0).sum())
        if negative:
            warnings.append(f"{negative:,} negative arrival values are invalid and will be excluded from modelling.")
        if frame[["commodity", "state"]].isna().any().any():
            warnings.append("Some rows have missing commodity or state identifiers and cannot be selected.")
        if "price_rs_qtl" not in frame.columns:
            warnings.append("price_rs_qtl is absent. The application will forecast arrivals without price as a predictor.")
        valid = dates.notna()
        key_frame = frame.loc[valid, ["commodity", "state"]].copy()
        key_frame["month"] = dates.loc[valid].dt.to_period("M").astype(str)
        dupes = int(key_frame.duplicated(KEY_COLUMNS).sum())
        if dupes:
            warnings.append(f"{dupes:,} duplicate commodity-state-month records exist. A selected series with duplicates is rejected rather than silently aggregated.")
    return {"errors": errors, "warnings": warnings}


def dataset_profile(frame: pd.DataFrame) -> dict:
    dates = pd.to_datetime(frame["month"], errors="coerce")
    arrivals = pd.to_numeric(frame["arrivals_t"], errors="coerce")
    keys = frame[["commodity", "state"]].copy()
    keys["month"] = dates.dt.to_period("M").astype(str)
    return {
        "rows": int(len(frame)),
        "commodities": int(frame["commodity"].nunique(dropna=True)),
        "states": int(frame["state"].nunique(dropna=True)),
        "missing_arrivals": int(arrivals.isna().sum()),
        "negative_arrivals": int((arrivals < 0).sum()),
        "date_min": str(dates.min().date()) if dates.notna().any() else "Unavailable",
        "date_max": str(dates.max().date()) if dates.notna().any() else "Unavailable",
        "duplicate_keys": int(keys.duplicated(KEY_COLUMNS).sum()),
        "columns": list(frame.columns),
    }


def prepare_monthly_series(subset: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Create a month-start indexed series; preserve missingness and reject duplicates."""
    data = subset.copy()
    data["month"] = pd.to_datetime(data["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    data["arrivals_t"] = pd.to_numeric(data["arrivals_t"], errors="coerce")
    data = data.dropna(subset=["month"])
    negative_count = int((data["arrivals_t"] < 0).sum())
    data.loc[data["arrivals_t"] < 0, "arrivals_t"] = np.nan
    if data["month"].duplicated().any():
        count = int(data["month"].duplicated(keep=False).sum())
        raise ValueError(f"This selected commodity-state series contains {count} rows in duplicated months. Resolve duplicates using the source documentation; the application will not guess how to aggregate them.")
    monthly = data.set_index("month")["arrivals_t"].sort_index()
    if monthly.empty:
        return monthly, {"calendar_months": 0, "observed_months": 0, "missing_months_or_values": 0, "missing_calendar_months_inserted": 0, "negative_values_excluded": negative_count}
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    inserted = len(full_index) - len(monthly)
    monthly = monthly.reindex(full_index)
    monthly.index.name = "month"
    diagnostics = {
        "calendar_months": int(len(monthly)),
        "observed_months": int(monthly.notna().sum()),
        "missing_months_or_values": int(monthly.isna().sum()),
        "missing_calendar_months_inserted": int(inserted),
        "negative_values_excluded": negative_count,
    }
    return monthly, diagnostics


def latest_contiguous_observations(series: pd.Series) -> tuple[pd.Series, dict]:
    """Return latest uninterrupted run; never bridge missing monthly reports."""
    values = series.sort_index()
    valid = np.flatnonzero(values.notna().to_numpy())
    if len(valid) == 0:
        return values.iloc[0:0], {"excluded_earlier_calendar_months": int(len(values)), "latest_contiguous_months": 0}
    last = int(valid[-1])
    start = last
    while start > 0 and pd.notna(values.iloc[start - 1]):
        if (values.index[start].to_period("M") - values.index[start - 1].to_period("M")).n != 1:
            break
        start -= 1
    segment = values.iloc[start:last + 1].astype(float)
    return segment, {
        "excluded_earlier_calendar_months": int(start),
        "latest_contiguous_months": int(len(segment)),
        "latest_contiguous_start": str(segment.index.min().date()),
        "latest_contiguous_end": str(segment.index.max().date()),
    }
