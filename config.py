"""
Central configuration for the AQI Predictor project.

This acts as the "feature store & model registry" abstraction layer.
By default it uses local Parquet files (zero-cost, zero-setup, fully
serverless-friendly since it lives on whatever runner executes the
pipeline). If you want a real managed feature store, swap the
`save_features` / `load_features` functions below for Hopsworks or
Vertex AI Feature Store calls -- the rest of the pipeline doesn't care
where the data lives.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# City configuration -- change this to forecast a different city.
# Defaults to Bahawalpur, Pakistan.
# ---------------------------------------------------------------------------
CITY_NAME = os.environ.get("AQI_CITY_NAME", "Bahawalpur")
LATITUDE = float(os.environ.get("AQI_LAT", 29.3956))
LONGITUDE = float(os.environ.get("AQI_LON", 71.6836))
TIMEZONE = os.environ.get("AQI_TZ", "auto")

# ---------------------------------------------------------------------------
# Paths (local stand-in for a managed feature store / model registry)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
FEATURE_STORE_DIR = ROOT / "feature_store"
MODEL_REGISTRY_DIR = ROOT / "models"
FEATURE_STORE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_REGISTRY_DIR.mkdir(exist_ok=True, parents=True)

FEATURES_PATH = FEATURE_STORE_DIR / "aqi_features.parquet"
METRICS_PATH = MODEL_REGISTRY_DIR / "metrics.json"

# Forecast horizons we train separate models for (in days ahead)
HORIZONS_DAYS = [1, 2, 3]

# AQI (US EPA scale) category breakpoints, used for alerting + coloring
AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def aqi_category(value: float):
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "Unknown", "#999999"
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= value <= hi:
            return label, color
    return "Hazardous", "#7e0023"


def save_features(df):
    """Persist the feature table. Swap this for Hopsworks/Vertex AI if desired."""
    df.to_parquet(FEATURES_PATH, index=False)


def load_features():
    import pandas as pd
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"No features found at {FEATURES_PATH}. Run feature_pipeline.py first."
        )
    return pd.read_parquet(FEATURES_PATH)
