# Project Specification and Acceptance Criteria

## Project identity

**Title:** AI-Based Agricultural Market Arrival Forecasting Using Machine Learning Techniques.

**Aim:** Design, develop and evaluate an AI-based agricultural market-arrival forecasting system using machine-learning techniques to predict future quantities of agricultural commodities arriving at selected wholesale markets.

**Implementation platform:** Streamlit. **Forecast target:** monthly arrival quantity (`arrivals_t`) in tonnes, according to the uploaded source's field definition.

## Objectives

1. Obtain and preprocess historical agricultural market-arrival data for selected commodities and geographies.
2. Identify relevant temporal and market-related features supported by the available data.
3. Develop forecasting models using ARIMA, XGBoost and LightGBM, and LSTM/GRU where data and dependencies permit.
4. Evaluate and compare models using MAE, RMSE and R² where appropriate.
5. Develop an interactive dashboard for historical analysis, forecasting and model comparison.

## Required workflow

- Ingest CSV and CSV.GZ files.
- Validate required fields, date parsing, numeric values, missingness, negative values and duplicate commodity-state-month keys.
- Filter one commodity and one state at a time; do not mix different commodities, geographies or incompatible units.
- Normalize dates to month-start timestamps and make missing calendar months explicit.
- Do not zero-fill missing reports or interpolate them.
- Reject unresolved duplicate observations rather than silently aggregating them.
- Model only the latest uninterrupted sequence of valid monthly observations.
- Explore historical arrivals and descriptive statistics.
- Compare naïve and seasonal-naïve baselines with ARIMA, XGBoost and LightGBM; LSTM/GRU are optional and dependency/data-length gated.
- Use chronological 70%/15%/15% training, validation and test partitions.
- Select by validation MAE; keep final test independent of model selection.
- Compare models on identical test dates and report MAE, RMSE and R² where meaningful.
- Generate nonnegative future arrival forecasts with monthly dates, charts and CSV downloads.
- Export run metadata for reproducibility.

## Academic boundaries

- Market arrivals are not direct consumer demand, sales, purchases, consumption, crop yield or commodity price.
- The candidate data are state-level monthly observations, not daily market-centre records.
- No fabricated data, metrics or empirical claims.
- No random time-series split, future leakage, silent zero-fill, interpolation or gap-bridging.
- Do not assume weather or production variables exist.
- Do not claim real-time updating without an implemented update mechanism.
- Do not generalise Indian market results to Nigeria without separate evidence.

## Acceptance criteria

- `streamlit run app.py` starts the dashboard in an environment where requirements are installed.
- CSV and CSV.GZ upload and schema validation work.
- Missing observations remain distinct from true zero arrivals.
- Duplicate monthly records are clearly rejected for manual resolution.
- Commodity/state selection and historical plots work.
- Modelling uses a contiguous monthly series and chronological partitions.
- Models are compared on the same final test dates; test data do not select the winner.
- Metrics derive from actual predictions only.
- Forecast dates are consecutive month-start dates after the final observation.
- Downloaded forecast, test predictions, comparison and run metadata correspond to displayed outputs.
- Automated tests pass.
