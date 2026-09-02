"""
Backfill historical (features, targets)
========================================
Open-Meteo's air-quality endpoint only keeps ~92 days of history behind
`past_days`, but it also supports explicit `start_date`/`end_date` for
its historical archive, which goes back much further. This script pages
through that archive in monthly chunks, runs the same feature
engineering as feature_pipeline.py, and writes the result into the
feature store -- giving you a real training set instead of just the
rolling 92-day window.

Usage:
    python backfill_pipeline.py --start 2024-01-01 --end 2025-08-01
"""
import argparse

import pandas as pd

import config
from feature_pipeline import (
    AIR_QUALITY_HOURLY_VARS,
    WEATHER_HOURLY_VARS,
    AIR_QUALITY_URL,
    WEATHER_URL,
    engineer_features,
    fetch_json,
)


def fetch_archive_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    common = dict(
        latitude=config.LATITUDE,
        longitude=config.LONGITUDE,
        timezone=config.TIMEZONE,
        start_date=start_date,
        end_date=end_date,
    )
    aq_json = fetch_json(AIR_QUALITY_URL, {**common, "hourly": ",".join(AIR_QUALITY_HOURLY_VARS)})
    wx_json = fetch_json(WEATHER_URL, {**common, "hourly": ",".join(WEATHER_HOURLY_VARS)})
    aq_df = pd.DataFrame(aq_json["hourly"])
    wx_df = pd.DataFrame(wx_json["hourly"])
    raw = aq_df.merge(wx_df, on="time", how="inner")
    raw["time"] = pd.to_datetime(raw["time"])
    return raw


def run(start: str, end: str, chunk_days: int = 30):
    date_range = pd.date_range(start, end, freq=f"{chunk_days}D")
    if date_range.empty or date_range[-1] < pd.Timestamp(end):
        date_range = date_range.append(pd.DatetimeIndex([pd.Timestamp(end)]))

    all_raw = []
    for i in range(len(date_range) - 1):
        chunk_start = date_range[i].strftime("%Y-%m-%d")
        chunk_end = min(date_range[i + 1], pd.Timestamp(end)).strftime("%Y-%m-%d")
        print(f"[backfill] fetching {chunk_start} -> {chunk_end}")
        all_raw.append(fetch_archive_chunk(chunk_start, chunk_end))

    raw = pd.concat(all_raw).drop_duplicates(subset="time").sort_values("time")
    features = engineer_features(raw)

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
    print(f"[backfill] Stored {len(combined)} total rows -> {config.FEATURES_PATH}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=30)
    args = parser.parse_args()
    run(args.start, args.end, args.chunk_days)
