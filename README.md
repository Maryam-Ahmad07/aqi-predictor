# AQI Predictor — Serverless 3-Day Air Quality Forecast

An end-to-end, 100%-serverless system that predicts a city's Air Quality
Index (AQI) 1, 2, and 3 days ahead: automated data collection → feature
engineering → model training → an interactive forecast dashboard.

Default city: **Karachi, Pakistan** (change via environment variables — see below).

## 1. Architecture

```
Weather & Pollution API (Open-Meteo, no key needed)
        │  raw data
        ▼
 feature_pipeline.py  ──features──▶  Feature Store (feature_store/aqi_features.parquet)
        ▲                                   │
        │ (hourly, via GitHub Actions)      │ features
        │                                   ▼
        │                          training_pipeline.py ──model──▶ Model Registry (models/)
        │ (daily, via GitHub Actions)                                    │
        │                                                                │
        └────────────────────────────────────────────────────  app.py (Streamlit dashboard)
                                                                  loads features + model,
                                                                  shows forecast + alerts
```

The "feature store" and "model registry" are local Parquet/joblib files
committed back to the repo by CI — genuinely serverless (no databases or
servers to run/pay for) and a drop-in swap for Hopsworks or Vertex AI
Feature Store if you want a managed version later (see `config.py`,
`save_features`/`load_features`).

## 2. Components

| File | Purpose |
|---|---|
| `config.py` | City config, paths, AQI category/alert thresholds |
| `feature_pipeline.py` | Fetches raw weather+pollutant data (past ~92 days + 3-day forecast), engineers features, upserts into the feature store. Runs hourly. |
| `backfill_pipeline.py` | Pulls Open-Meteo's longer historical archive in chunks for a bigger training set (beyond the rolling 92-day window). Run once (or periodically) to build up history. |
| `training_pipeline.py` | Trains Ridge, Random Forest, Gradient Boosting, and a small MLP per horizon; picks the best by RMSE; evaluates with RMSE/MAE/R²; saves the winning model + a SHAP explainability plot. Runs daily. |
| `app.py` | Streamlit dashboard: 3-day forecast cards with color-coded AQI categories, hazard alerts, historical trend/EDA charts, model performance table, SHAP feature-importance plot. |
| `.github/workflows/*.yml` | CI/CD: feature pipeline hourly, training pipeline daily, auto-committing results. |
| `tests/generate_synthetic_data.py` | **Dev/demo only** — fabricates a realistic dataset so you can test everything without waiting on real historical data (see note below). |

## 3. Features engineered

- **Time-based:** hour, day, month, day-of-week, weekend flag, cyclical
  sin/cos encodings of hour and month.
- **Pollutant/AQI dynamics:** 24h lag, 24h rolling mean/std, and 1h/24h
  **AQI change rate** for AQI, PM2.5, PM10, ozone, and NO₂.
- **Raw weather:** temperature, humidity, dew point, precipitation,
  pressure, wind speed/direction, cloud cover.
- **Targets:** `target_aqi_1d`, `target_aqi_2d`, `target_aqi_3d` — US AQI
  24/48/72 hours ahead of each row.

## 4. Models & evaluation

Four candidates per horizon — Ridge Regression, Random Forest, Gradient
Boosting, and an MLP neural net — trained on a **time-ordered** (never
shuffled) 80/20 split, so evaluation reflects genuine forecasting
performance rather than leaking future information into training. The
best model per horizon (lowest test RMSE) is kept.

**Results on the validation dataset used to build/test this repo** (see
note below on data source):

| Horizon | Best model | RMSE | MAE | R² |
|---|---|---|---|---|
| +1 day | Ridge | 4.03 | 3.25 | 0.939 |
| +2 days | Ridge | 4.20 | 3.42 | 0.934 |
| +3 days | Ridge | 4.09 | 3.30 | 0.938 |

Ridge Regression won on this dataset — a good reminder that a simple,
well-regularized linear model on rich engineered features (lags, rolling
stats, cyclical time encodings) can beat more complex models,
especially with a moderate amount of training data. As more real history
accumulates via `backfill_pipeline.py`, it's worth re-checking whether
Random Forest/Gradient Boosting pull ahead.

Feature importance (SHAP) consistently shows the AQI lag/rolling-mean
features and time-of-day encoding as the dominant predictors, which
matches physical intuition: AQI is highly autocorrelated and has strong
diurnal traffic-driven patterns.

## 5. Alerts

The dashboard flags any forecasted day with **AQI ≥ 151** (EPA's
"Unhealthy" threshold) with an on-screen warning naming the day, value,
and category, so sensitive groups know to limit outdoor exposure.

## 6. Running it yourself

```bash
pip install -r requirements.txt

# 1. Build up a feature store (real data — needs outbound internet to open-meteo.com)
python feature_pipeline.py --past-days 92 --forecast-days 3
python backfill_pipeline.py --start 2024-01-01 --end 2025-08-01   # optional, more history

# 2. Train models
python training_pipeline.py

# 3. Launch the dashboard
streamlit run app.py
```

To point at a different city, set environment variables before running:
```bash
export AQI_CITY_NAME="Lahore"
export AQI_LAT=31.5497
export AQI_LON=74.3436
```

## 7. Deployment (serverless)

- **Feature/training pipelines:** the included GitHub Actions workflows
  (`.github/workflows/feature_pipeline.yml`, `training_pipeline.yml`)
  run on GitHub's free hosted runners — no server to manage. They
  commit updated Parquet/joblib files back to the repo, which doubles
  as a zero-cost feature store + model registry.
- **Dashboard:** push this repo to GitHub and deploy `app.py` for free
  on [Streamlit Community Cloud](https://streamlit.io/cloud) — it will
  auto-redeploy whenever CI commits new features/models.
- **Optional upgrade path:** swap `config.save_features`/`load_features`
  for Hopsworks or Vertex AI Feature Store calls, and swap the
  `joblib.dump`/`load` calls in `training_pipeline.py`/`app.py` for
  their model registries, without touching any other logic.

## 8. Honest limitations / what a real deployment still needs

- **This sandbox has no outbound internet to `open-meteo.com`**, so the
  metrics above and the SHAP plot were generated against
  `tests/generate_synthetic_data.py` — a hand-built synthetic dataset
  with realistic diurnal/seasonal AQI patterns, used purely to prove the
  pipeline runs correctly end-to-end. **Run `feature_pipeline.py` for
  real** (on your machine or in CI, where outbound internet is
  unrestricted) before trusting the numbers for an actual city.
- Open-Meteo's air-quality forecast itself is a CAMS-model-based
  forecast, not ground truth — like any AQI forecasting project, your
  model is learning to track a modeled/measured signal that has its own
  uncertainty.
- No LIME explanations were added (SHAP was prioritized since it has
  stronger theoretical guarantees and works cleanly with both tree and
  non-tree models here) — straightforward to add via the `lime` package
  if you want both.
- Deep learning (LSTM/Temporal Fusion Transformer, etc.) was
  intentionally left out of the default candidate set to keep the
  "serverless," free-tier-friendly promise (no GPU, fast CI runs); the
  MLP is included as a lightweight neural baseline. Swap in a
  Keras/PyTorch model in `candidate_models()` if you want to push
  accuracy further with more data.
