"""
FOR LOCAL TESTING/DEMO ONLY.

The sandbox this was built in cannot reach api.open-meteo.com, so this
script fabricates a realistic-looking raw hourly dataset (same schema
Open-Meteo returns) so the rest of the pipeline (feature engineering,
training, dashboard) can be exercised end-to-end. Real deployments
should never call this -- they use feature_pipeline.py against the
live API.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from feature_pipeline import engineer_features


def generate(n_hours=92 * 24 + 24, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.now(tz=None).floor("h") - pd.Timedelta(hours=n_hours)
    times = pd.date_range(start, periods=n_hours, freq="h")

    hours = times.hour.values
    doy = times.dayofyear.values

    # Diurnal + seasonal pattern typical of a polluted coastal city (traffic
    # peaks morning/evening, higher AQI in cooler months, some noise + a mild
    # upward trend to make forecasting non-trivial).
    diurnal = 20 * np.sin((hours - 8) / 24 * 2 * np.pi) + 15 * np.exp(-((hours - 9) ** 2) / 8) + 15 * np.exp(-((hours - 20) ** 2) / 8)
    seasonal = 25 * np.cos((doy - 15) / 365 * 2 * np.pi)
    trend = np.linspace(0, 8, n_hours)
    noise = rng.normal(0, 8, n_hours)
    ar_noise = pd.Series(noise).ewm(span=6).mean().values  # autocorrelated noise

    base_aqi = 95 + diurnal + seasonal + trend + ar_noise
    us_aqi = np.clip(base_aqi, 15, 400)

    pm2_5 = np.clip(us_aqi * 0.55 + rng.normal(0, 4, n_hours), 2, None)
    pm10 = np.clip(pm2_5 * 1.6 + rng.normal(0, 6, n_hours), 5, None)
    co = np.clip(300 + us_aqi * 2 + rng.normal(0, 50, n_hours), 100, None)
    no2 = np.clip(15 + us_aqi * 0.2 + rng.normal(0, 5, n_hours), 1, None)
    so2 = np.clip(8 + us_aqi * 0.1 + rng.normal(0, 3, n_hours), 0, None)
    ozone = np.clip(40 - us_aqi * 0.05 + rng.normal(0, 8, n_hours), 1, None)

    temp = 26 + 6 * np.sin((hours - 15) / 24 * 2 * np.pi) + 6 * np.cos((doy - 200) / 365 * 2 * np.pi) + rng.normal(0, 1.5, n_hours)
    humidity = np.clip(60 - 15 * np.sin((hours - 15) / 24 * 2 * np.pi) + rng.normal(0, 5, n_hours), 10, 100)
    dew_point = temp - (100 - humidity) / 5
    precip = np.clip(rng.exponential(0.05, n_hours) - 0.08, 0, None)
    pressure = 1012 + 3 * np.cos((doy) / 365 * 2 * np.pi) + rng.normal(0, 1, n_hours)
    wind_speed = np.clip(10 + 5 * np.sin((hours - 13) / 24 * 2 * np.pi) + rng.normal(0, 2, n_hours), 0, None)
    wind_dir = (180 + 60 * np.sin(doy / 365 * 2 * np.pi) + rng.normal(0, 20, n_hours)) % 360
    cloud_cover = np.clip(40 + rng.normal(0, 20, n_hours), 0, 100)

    raw = pd.DataFrame({
        "time": times,
        "pm10": pm10, "pm2_5": pm2_5, "carbon_monoxide": co,
        "nitrogen_dioxide": no2, "sulphur_dioxide": so2, "ozone": ozone,
        "us_aqi": us_aqi,
        "temperature_2m": temp, "relative_humidity_2m": humidity,
        "dew_point_2m": dew_point, "precipitation": precip,
        "surface_pressure": pressure, "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_dir, "cloud_cover": cloud_cover,
    })
    return raw


if __name__ == "__main__":
    raw = generate()
    features = engineer_features(raw)
    config.save_features(features)
    print(f"Synthetic dataset written: {len(features)} rows -> {config.FEATURES_PATH}")
