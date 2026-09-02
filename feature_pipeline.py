"""
Feature pipeline
================
1. Fetches raw hourly weather + pollutant data from Open-Meteo
   (https://open-meteo.com) -- no API key required, generous free tier,
   and it conveniently exposes both recent history (`past_days`) and a
   forecast (`forecast_days`) from the same endpoint, which is exactly
   what a "3-day-ahead AQI forecaster" needs.
2. Computes model-ready features: time-based features (hour/day/month/
   day-of-week, cyclical encodings), lag features, rolling averages, and
   the AQI change-rate.
3. Upserts the resulting rows into the local Parquet "feature store"
   (see config.py) -- swap that layer for Hopsworks/Vertex AI in
   production without touching anything below.

Run standalone:
    python feature_pipeline.py                # last 92 days + 3-day forecast
    python feature_pipeline.py --past-days 7   # smaller/faster pull

This same script is what GitHub Actions calls every hour (see
.github/workflows/feature_pipeline.yml) and what backfill_pipeline.py
calls repeatedly to build historical training data.
"""
import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_HOURLY_VARS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi",
]
WEATHER_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "surface_pressure", "wind_speed_10m",
    "wind_direction_10m", "cloud_cover",
]


def fetch_json(url, params, retries=3):
    last_err = None
    for _ in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_raw_data(past_days: int, forecast_days: int):
    """Step 1: fetch raw weather + pollutant data from the external APIs."""
    common = dict(
        latitude=config.LATITUDE,
        longitude=config.LONGITUDE,
        timezone=config.TIMEZONE,
        past_days=past_days,
        forecast_days=forecast_days,
    )

    aq_json = fetch_json(AIR_QUALITY_URL, {
        **common,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
    })
    wx_json = fetch_json(WEATHER_URL, {
        **common,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
    })

    aq_df = pd.DataFrame(aq_json["hourly"])
    wx_df = pd.DataFrame(wx_json["hourly"])
    raw = aq_df.merge(wx_df, on="time", how="inner")
    raw["time"] = pd.to_datetime(raw["time"])
    return raw


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Step 2: compute model inputs (features) and outputs (targets)."""
    df = raw.sort_values("time").reset_index(drop=True).copy()

    # --- time-based features -------------------------------------------------
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # cyclical encodings so the model understands hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- lag / rolling / change-rate features on AQI and pollutants ---------
    for col in ["us_aqi", "pm2_5", "pm10", "ozone", "nitrogen_dioxide"]:
        df[f"{col}_lag24"] = df[col].shift(24)
        df[f"{col}_roll24_mean"] = df[col].rolling(24, min_periods=6).mean()
        df[f"{col}_roll24_std"] = df[col].rolling(24, min_periods=6).std()

    # AQI change rate = how fast AQI is rising/falling right now
    df["aqi_change_rate_1h"] = df["us_aqi"].diff(1)
    df["aqi_change_rate_24h"] = df["us_aqi"].diff(24)

    # --- forecast targets: AQI 24h / 48h / 72h into the future ---------------
    for h in config.HORIZONS_DAYS:
        df[f"target_aqi_{h}d"] = df["us_aqi"].shift(-24 * h)

    df["city"] = config.CITY_NAME
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


def run(past_days: int = 92, forecast_days: int = 3, upsert: bool = True) -> pd.DataFrame:
    print(f"[feature_pipeline] Fetching {past_days}d history + {forecast_days}d forecast "
          f"for {config.CITY_NAME} ({config.LATITUDE}, {config.LONGITUDE})")
    raw = fetch_raw_data(past_days=past_days, forecast_days=forecast_days)
    features = engineer_features(raw)

    if upsert:
        try:
            existing = config.load_features()
            combined = (
                pd.concat([existing, features])
                .drop_duplicates(subset="time", keep="last")
                .sort_values("time")
                .reset_index(drop=True)
            )
        except FileNotFoundError:
            combined = features
        config.save_features(combined)
        print(f"[feature_pipeline] Stored {len(combined)} total rows "
              f"({len(features)} fetched this run) -> {config.FEATURES_PATH}")
        return combined

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--past-days", type=int, default=92,
                         help="Days of history to pull (Open-Meteo max ~92 for air quality)")
    parser.add_argument("--forecast-days", type=int, default=3)
    args = parser.parse_args()
    try:
        run(past_days=args.past_days, forecast_days=args.forecast_days)
    except RuntimeError as e:
        print(f"[feature_pipeline] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
