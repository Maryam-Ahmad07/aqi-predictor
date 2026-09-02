"""
Training pipeline
==================
1. Fetches historical (features, targets) from the feature store.
2. For each forecast horizon (1d/2d/3d ahead), trains several candidate
   models -- Ridge Regression, Random Forest, Gradient Boosting, and an
   MLP (small neural net) -- and picks the best by validation RMSE.
3. Evaluates with RMSE, MAE and R^2 on a held-out, time-ordered test
   split (never shuffled -- this is a forecasting problem).
4. Saves the winning model per horizon to the local model registry
   (models/), plus metrics.json and a SHAP summary plot for the 1-day
   model so you can explain *why* it predicted what it predicted.

Run standalone:
    python training_pipeline.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = None  # populated dynamically, excludes ids/targets/time


def build_feature_columns(df: pd.DataFrame):
    exclude = {"time", "city", "fetched_at"} | {f"target_aqi_{h}d" for h in config.HORIZONS_DAYS}
    return [c for c in df.columns if c not in exclude]


def time_ordered_split(df: pd.DataFrame, test_frac: float = 0.2):
    n_test = int(len(df) * test_frac)
    return df.iloc[:-n_test], df.iloc[-n_test:]


def candidate_models():
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42,
            early_stopping=True,
        ),
    }


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_horizon(df: pd.DataFrame, horizon: int, feature_cols: list[str]):
    target_col = f"target_aqi_{horizon}d"
    data = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    if len(data) < 50:
        raise ValueError(
            f"Not enough clean rows ({len(data)}) to train the {horizon}d model. "
            "Run backfill_pipeline.py to gather more history."
        )

    train_df, test_df = time_ordered_split(data)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    fitted = {}
    for name, model in candidate_models().items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        results[name] = evaluate(y_test, preds)
        fitted[name] = model

    best_name = min(results, key=lambda n: results[n]["rmse"])
    print(f"  [{horizon}d] best model: {best_name} -> {results[best_name]}")

    bundle = {
        "model": fitted[best_name],
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_name": best_name,
        "horizon_days": horizon,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    joblib.dump(bundle, config.MODEL_REGISTRY_DIR / f"model_{horizon}d.joblib")
    return best_name, results


def shap_summary_plot(df: pd.DataFrame, feature_cols: list[str], horizon: int = 1):
    """Generate a SHAP feature-importance plot for the given horizon's model."""
    import shap

    bundle = joblib.load(config.MODEL_REGISTRY_DIR / f"model_{horizon}d.joblib")
    model, scaler = bundle["model"], bundle["scaler"]
    target_col = f"target_aqi_{horizon}d"
    data = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    sample = data[feature_cols].sample(min(200, len(data)), random_state=42)
    sample_s = scaler.transform(sample)

    if hasattr(model, "estimators_") or hasattr(model, "tree_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_s)
    else:
        background = shap.sample(sample_s, min(50, len(sample_s)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(sample_s, nsamples=100)

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False, max_display=12)
    plt.tight_layout()
    out_path = config.MODEL_REGISTRY_DIR / f"shap_summary_{horizon}d.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  SHAP summary plot saved -> {out_path}")


def run():
    print("[training_pipeline] Loading features from feature store...")
    df = config.load_features()
    feature_cols = build_feature_columns(df)
    print(f"[training_pipeline] {len(df)} rows, {len(feature_cols)} candidate features")

    all_metrics = {}
    for horizon in config.HORIZONS_DAYS:
        print(f"[training_pipeline] Training {horizon}-day-ahead model...")
        best_name, results = train_horizon(df, horizon, feature_cols)
        all_metrics[f"{horizon}d"] = {"best_model": best_name, "results": results}

    try:
        shap_summary_plot(df, feature_cols, horizon=1)
    except Exception as e:  # noqa: BLE001
        print(f"[training_pipeline] SHAP plot skipped: {e}")

    with open(config.METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[training_pipeline] Metrics written -> {config.METRICS_PATH}")
    return all_metrics


if __name__ == "__main__":
    run()
