# AI-Based Agricultural Market Arrival Forecasting Using Machine Learning Techniques

A complete, modular **Streamlit research application** for analysing and forecasting monthly agricultural market-arrival quantities from a user-supplied historical CSV/CSV.GZ dataset.

## 1. Forecasting target and academic boundary

The forecast target is `arrivals_t`: reported quantity of agricultural produce arriving at the selected wholesale-market geography, represented in tonnes by the candidate dataset. **Market arrivals are not direct measurements of consumer demand, sales, purchases, consumption, crop yield or commodity prices.** The candidate dataset is state-level and monthly, not daily market-centre data. Do not generalise results from Indian markets to Nigerian markets without separate evidence.

## 2. Features included

- CSV and gzip-compressed CSV upload.
- Required-column, date, numeric, missing-value, negative-value and duplicate-key validation.
- Commodity and state filtering without mixing unlike series.
- Monthly calendar construction and missing-report diagnostics.
- Explicit rejection of unresolved duplicate commodity-state-month observations.
- Latest uninterrupted observation segment for modelling; no missing-value zero-fill, interpolation or artificial adjacency.
- Historical time-series chart and descriptive statistics.
- Naïve, Seasonal Naïve and ARIMA models.
- Optional XGBoost and LightGBM models.
- Optional LSTM and GRU models when TensorFlow is installed and at least 48 contiguous observations exist.
- Chronological 70%/15%/15% train/validation/test split.
- Validation-based model selection; final test set remains separate.
- MAE, RMSE and R² where appropriate, test prediction tables and model comparison.
- Future forecast charts and CSV export.
- Export of run metadata including file SHA-256, selection, horizon and training period.
- Automated unit tests.

## 3. Installation and running

Use Python 3.10–3.12 for the broadest compatibility with the optional machine-learning libraries. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The standard installation includes Streamlit, Pandas, NumPy, scikit-learn, statsmodels, Plotly, XGBoost and LightGBM. The optional tree requirements file is also provided for users who prefer to install those models separately:

```bash
pip install -r requirements-optional-tree.txt
```

LSTM and GRU require TensorFlow. TensorFlow support is platform/Python-version dependent; install only in a compatible environment:

```bash
pip install -r requirements-sequence.txt
```

The dashboard detects optional libraries and only offers models whose dependencies are available. It does not require AppDeploy.

## 4. Dataset format

Upload a `.csv` or `.csv.gz` file with these required column names:

| Column | Meaning |
|---|---|
| `month` | Date or month value parseable by Pandas |
| `commodity` | Commodity identifier/name |
| `state` | State/geographic identifier |
| `arrivals_t` | Arrival quantity in tonnes, as defined by the source |

Other columns are retained in the uploaded dataframe but are not automatically used as predictors. In particular, price is not treated as an arrival predictor. Verify units and definitions against the source documentation before analysis.

## 5. Methodology and safeguards

1. Validate the input schema and parse dates and numeric arrival values.
2. Convert dates to month-start timestamps.
3. Reject duplicated months within a selected commodity-state series; do not silently sum records with uncertain meaning.
4. Insert absent calendar months as missing values. Missing reports are not assumed to mean zero arrivals.
5. Use only the latest uninterrupted segment of valid monthly observations for training. Earlier segments are excluded rather than joined across gaps.
6. Split observations chronologically: 70% training, 15% validation and 15% final test. No random shuffling is used.
7. Select the candidate with the lowest validation MAE. Refit it on training plus validation before final test evaluation. The final test is not used to select the model.
8. Compare all successful candidates on the same test period using MAE, RMSE and R² where appropriate.
9. Train on historical arrivals and recursively generate forecasts. Predicted negative arrival quantities are clipped to zero and this should be disclosed in the research methodology.

Model configurations are fixed baseline configurations, not the result of exhaustive hyperparameter tuning. A model appearing in the dashboard does not prove that it performs well. Report only actual metrics produced from the chosen real dataset.

## 6. Dataset provenance

The planned primary source is the APMC Arrivals and Prices dataset published through India's Agricultural Marketing Information Network (AGMARKNET), accessed through India Data Portal or the documented AGMARKNET distribution. The ZIP does not include or fabricate a research dataset. Preserve the downloaded source file and record its exact URL, retrieval date, licence/terms, checksum, coverage and all transformations in the final report.

## 7. Tests

From the project root:

```bash
python -m pytest -q
python -m compileall -q .
```

Tests cover schema validation, missing values, missing calendar months, duplicate rejection, negative values, contiguous-series handling, chronological splitting and metric calculations.

## 8. Project structure

```text
agricultural_market_arrival_forecasting/
├── app.py
├── README.md
├── PROJECT_SPECIFICATION.md
├── requirements.txt
├── requirements-optional-tree.txt
├── requirements-sequence.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── plots.py
└── tests/
    ├── test_data.py
    └── test_evaluation.py
```

## 9. Limitations

This is an offline research application, not a real-time market service. It has no automatic source refresh and no verified weather, production or other external predictors. State-level monthly data limits geographic and temporal detail. Forecast performance depends on data quality, series length and structural changes. No empirical results are asserted in advance.
