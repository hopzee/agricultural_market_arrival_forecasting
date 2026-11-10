"""Streamlit interface for AI-based agricultural market-arrival forecasting."""
from datetime import datetime, timezone
import hashlib
import json
import streamlit as st
import pandas as pd
from src.data import load_dataset, validate_dataset, dataset_profile, prepare_monthly_series, latest_contiguous_observations
from src.models import available_models, fit_predict
from src.evaluation import chronological_three_way_split, evaluate_candidates
from src.plots import arrivals_chart, forecast_chart, comparison_chart

st.set_page_config(page_title="Agricultural Market Arrival Forecasting", page_icon="🌾", layout="wide")
st.title("🌾 AI-Based Agricultural Market Arrival Forecasting Using Machine Learning Techniques")
st.caption("Research application | Monthly state-level market-arrival forecasting")
st.info("**Forecast target:** reported agricultural market arrivals (tonnes). Market arrivals are not direct measurements of consumer demand, sales, consumption, crop yield or commodity prices.")

with st.sidebar:
    st.header("1. Dataset")
    uploaded = st.file_uploader("Upload AGMARKNET-derived CSV or CSV.GZ", type=["csv", "gz"])
    st.caption("Required fields: month, commodity, state, arrivals_t")
    st.divider()
    st.header("2. Forecast settings")
    horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=24, value=6, step=1)

if uploaded is None:
    st.markdown("### Getting started")
    st.write("Upload your historical dataset to begin. The application does not generate or insert fake research records.")
    st.markdown("""**Workflow**
    1. Upload and validate the dataset.
    2. Select one commodity and one state.
    3. Inspect historical arrivals and reporting gaps.
    4. Evaluate models using chronological training, validation and test periods.
    5. Generate and download future arrival forecasts and evaluation results.
    """)
    st.stop()

try:
    raw = load_dataset(uploaded)
    validation = validate_dataset(raw)
except Exception as exc:
    st.error(f"The uploaded file could not be read: {exc}")
    st.stop()
if validation["errors"]:
    st.error("The dataset cannot be processed until these issues are resolved:")
    for message in validation["errors"]:
        st.write(f"- {message}")
    st.stop()

profile = dataset_profile(raw)
metric_cols = st.columns(5)
metric_cols[0].metric("Records", f"{profile['rows']:,}")
metric_cols[1].metric("Commodities", f"{profile['commodities']:,}")
metric_cols[2].metric("States", f"{profile['states']:,}")
metric_cols[3].metric("Missing arrivals", f"{profile['missing_arrivals']:,}")
metric_cols[4].metric("Duplicate keys", f"{profile['duplicate_keys']:,}")
with st.expander("Dataset validation, coverage and provenance", expanded=True):
    st.write("Columns detected:", profile["columns"])
    st.write(f"Date coverage: {profile['date_min']} to {profile['date_max']}")
    st.write(f"Negative arrival values: {profile['negative_arrivals']:,}")
    st.write("Arrival unit: tonnes, as represented by the selected dataset's arrivals_t field. Verify this against the original source metadata.")
    for warning in validation["warnings"]:
        st.warning(warning)
    if profile["duplicate_keys"]:
        st.warning("Duplicate keys exist in the uploaded dataset. Any selected commodity-state series containing duplicate months will be stopped for manual resolution.")

commodities = sorted(raw["commodity"].dropna().astype(str).unique())
states = sorted(raw["state"].dropna().astype(str).unique())
if not commodities or not states:
    st.error("No valid commodity or state values are available.")
    st.stop()
col_a, col_b = st.columns(2)
commodity = col_a.selectbox("Select commodity", commodities)
state = col_b.selectbox("Select state", states)
selected = raw[(raw["commodity"].astype(str) == commodity) & (raw["state"].astype(str) == state)].copy()
try:
    full_series, diagnostics = prepare_monthly_series(selected)
except Exception as exc:
    st.error(f"Selected series cannot be prepared: {exc}")
    st.stop()
if full_series.empty:
    st.warning("No valid dated observations exist for this commodity and state.")
    st.stop()

st.header("3. Historical market-arrival analysis")
st.plotly_chart(arrivals_chart(full_series, commodity, state), use_container_width=True)
observed = full_series.dropna()
if not observed.empty:
    stats = st.columns(4)
    stats[0].metric("Observed months", f"{len(observed):,}")
    stats[1].metric("Mean arrivals (tonnes)", f"{observed.mean():,.2f}")
    stats[2].metric("Median arrivals (tonnes)", f"{observed.median():,.2f}")
    stats[3].metric("Maximum arrivals (tonnes)", f"{observed.max():,.2f}")
st.caption("Missing calendar months and missing arrival values remain missing. They are not converted to zero or interpolated.")
with st.expander("Monthly data and reporting diagnostics"):
    st.json(diagnostics)
    st.dataframe(full_series.rename("arrivals_t").rename_axis("month").reset_index(), use_container_width=True)

series, continuity = latest_contiguous_observations(full_series)
if continuity.get("excluded_earlier_calendar_months", 0):
    st.warning(f"Modelling uses only the latest uninterrupted segment ({continuity.get('latest_contiguous_start', 'unknown')} to {continuity.get('latest_contiguous_end', 'unknown')}). Earlier calendar months excluded: {continuity['excluded_earlier_calendar_months']}. No missing values were imputed.")
if len(series) < 24:
    st.warning(f"The latest uninterrupted segment contains {len(series)} months. At least 24 are required for this application's three-way chronological evaluation. No performance results will be fabricated.")
    st.stop()

try:
    train, validation_set, test = chronological_three_way_split(series)
except ValueError as exc:
    st.warning(str(exc)); st.stop()

st.header("4. Model training and evaluation")
st.write(f"Chronological split: **{len(train)} training**, **{len(validation_set)} validation**, and **{len(test)} final test** months. The final test period is not used to select the winning model.")
model_list = available_models()
st.caption("Available models are determined by installed dependencies. LSTM and GRU require TensorFlow and at least 48 contiguous monthly observations.")
with st.spinner("Training and evaluating available models. This may take a little time..."):
    results, test_predictions = evaluate_candidates(model_list, train, validation_set, test, fit_predict)
st.dataframe(results, use_container_width=True)
completed = results[(results["Status"] == "Completed") & results["Validation MAE"].notna()]
if completed.empty:
    st.error("No model completed evaluation. Review the status messages and install any required optional dependencies.")
else:
    winner = completed.sort_values("Validation MAE").iloc[0]
    st.success(f"Best candidate by validation MAE: {winner['Model']}. This selection is based on validation data, not final test performance.")
    st.plotly_chart(comparison_chart(results, y="Test MAE"), use_container_width=True)
    st.subheader("Final test predictions")
    prediction_model = st.selectbox("View test predictions for", list(test_predictions.keys()))
    st.dataframe(test_predictions[prediction_model], use_container_width=True)
    st.download_button("Download selected test predictions CSV", test_predictions[prediction_model].to_csv(index=False).encode("utf-8"),
        file_name=f"test_predictions_{prediction_model.lower().replace(' ', '_')}.csv", mime="text/csv")

st.download_button("Download model comparison CSV", results.to_csv(index=False).encode("utf-8"),
    file_name="agricultural_arrival_model_comparison.csv", mime="text/csv")

st.header("5. Generate future market-arrival forecast")
forecastable = [name for name in model_list if name in model_list]
default_index = forecastable.index(winner["Model"]) if not completed.empty and winner["Model"] in forecastable else 0
chosen_model = st.selectbox("Forecasting model", forecastable, index=default_index)
if st.button("Generate future forecast", type="primary"):
    try:
        future = fit_predict(chosen_model, series, int(horizon))
        st.plotly_chart(forecast_chart(series, future, commodity, state), use_container_width=True)
        forecast_frame = future.rename_axis("month").rename("forecast_arrivals_t").reset_index()
        st.dataframe(forecast_frame, use_container_width=True)
        st.download_button("Download future forecast CSV", forecast_frame.to_csv(index=False).encode("utf-8"),
            file_name="agricultural_market_arrival_forecast.csv", mime="text/csv")
        run_metadata = {"project_title": "AI-Based Agricultural Market Arrival Forecasting Using Machine Learning Techniques",
            "commodity": commodity, "state": state, "forecast_horizon_months": int(horizon),
            "model": chosen_model, "training_observations": int(len(series)),
            "training_start": str(series.index.min().date()), "training_end": str(series.index.max().date()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "uploaded_file_sha256": hashlib.sha256(uploaded.getvalue()).hexdigest()}
        st.download_button("Download forecast run metadata JSON", json.dumps(run_metadata, indent=2).encode("utf-8"),
            file_name="forecast_run_metadata.json", mime="application/json")
    except Exception as exc:
        st.error(f"Forecast could not be generated: {exc}")

st.divider()
st.caption("Research prototype. Forecasts estimate historical reported market-arrival quantities; they are not guaranteed outcomes, real-time updates, or direct measures of consumer demand. Record the exact dataset source, retrieval date, licence and unit definitions in the research report.")
