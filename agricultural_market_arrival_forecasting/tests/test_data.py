import pandas as pd
import pytest
from src.data import validate_dataset, prepare_monthly_series, latest_contiguous_observations


def sample(rows=None):
    rows = rows or [("2024-01-01", 10), ("2024-02-01", 20)]
    return pd.DataFrame({"month": [r[0] for r in rows], "commodity": ["Maize"]*len(rows),
        "state": ["State A"]*len(rows), "arrivals_t": [r[1] for r in rows]})


def test_required_columns_validation():
    assert validate_dataset(sample())["errors"] == []


def test_missing_arrivals_are_not_zero_filled():
    series, _ = prepare_monthly_series(sample([("2024-01-01", 10), ("2024-02-01", None)]))
    assert pd.isna(series.iloc[1])


def test_missing_calendar_month_is_inserted_as_missing():
    series, diagnostics = prepare_monthly_series(sample([("2024-01-01", 10), ("2024-03-01", 30)]))
    assert len(series) == 3 and pd.isna(series.iloc[1])
    assert diagnostics["missing_calendar_months_inserted"] == 1


def test_duplicate_months_are_rejected_not_summed():
    with pytest.raises(ValueError, match="duplicated months"):
        prepare_monthly_series(sample([("2024-01-01", 10), ("2024-01-20", 20)]))


def test_latest_contiguous_segment_does_not_bridge_gap():
    idx = pd.date_range("2023-01-01", periods=6, freq="MS")
    segment, info = latest_contiguous_observations(pd.Series([1, 2, None, 4, 5, 6], index=idx))
    assert segment.tolist() == [4.0, 5.0, 6.0]
    assert info["excluded_earlier_calendar_months"] == 3


def test_negative_arrivals_are_excluded():
    series, diagnostics = prepare_monthly_series(sample([("2024-01-01", 10), ("2024-02-01", -5)]))
    assert pd.isna(series.iloc[1]) and diagnostics["negative_values_excluded"] == 1


def test_invalid_required_schema_is_reported():
    report = validate_dataset(pd.DataFrame({"date": ["2024-01"], "qty": [1]}))
    assert report["errors"]


def test_profile_counts_duplicate_keys():
    from src.data import dataset_profile
    df = sample([("2024-01-01", 10), ("2024-01-15", 12)])
    assert dataset_profile(df)["duplicate_keys"] == 1


def test_csv_gzip_loading():
    import gzip
    from io import BytesIO
    from src.data import load_dataset
    payload = b"month,commodity,state,arrivals_t\n2024-01,Maize,State A,12\n"
    class Upload(BytesIO):
        name = "arrivals.csv.gz"
    upload = Upload(gzip.compress(payload))
    frame = load_dataset(upload)
    assert list(frame.columns) == ["month", "commodity", "state", "arrivals_t"]
    assert frame.loc[0, "commodity"] == "Maize"
