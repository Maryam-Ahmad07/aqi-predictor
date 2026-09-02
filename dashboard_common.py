from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

import config


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, rgba(90, 166, 255, 0.14), transparent 25%),
                            radial-gradient(circle at bottom right, rgba(22, 163, 74, 0.10), transparent 25%),
                            linear-gradient(135deg, #071926 0%, #0d1f2d 30%, #10253a 100%);
                color: #edf7ff;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            div[data-testid="stHeader"] {
                background: rgba(7, 18, 26, 0.25);
                backdrop-filter: blur(10px);
            }
            .hero-panel {
                background: linear-gradient(135deg, rgba(18, 73, 102, 0.95), rgba(8, 26, 38, 0.9));
                border: 1px solid rgba(148, 213, 255, 0.25);
                border-radius: 26px;
                padding: 1.5rem 1.5rem 1.25rem;
                margin-bottom: 1.2rem;
                box-shadow: 0 20px 40px rgba(2, 9, 19, 0.32);
                position: relative;
                overflow: hidden;
            }
            .hero-panel::after {
                content: "";
                position: absolute;
                inset: 0 auto auto 0;
                width: 220px;
                height: 220px;
                background: radial-gradient(circle, rgba(125, 211, 252, 0.18), transparent 65%);
                pointer-events: none;
            }
            .city-pill {
                display: inline-block;
                padding: 0.5rem 0.9rem;
                border-radius: 999px;
                background: rgba(96, 165, 250, 0.12);
                border: 1px solid rgba(147, 197, 253, 0.42);
                color: #dfeeff;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-size: 0.7rem;
            }
            .summary-card {
                background: rgba(12, 27, 41, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 18px;
                padding: 1rem 1rem 0.8rem;
                box-shadow: 0 14px 28px rgba(2, 8, 15, 0.18);
                min-height: 120px;
            }
            .summary-label {
                color: #9cc9ff;
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            }
            .summary-value {
                color: white;
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
            }
            .summary-subtext {
                color: #c3d7ea;
                font-size: 0.82rem;
                margin-top: 0.35rem;
            }
            .forecast-card {
                border-radius: 22px;
                padding: 1.3rem 0.9rem 1.1rem;
                text-align: center;
                box-shadow: 0 14px 30px rgba(0,0,0,0.22);
                border: 1px solid rgba(255,255,255,0.10);
                background: rgba(15, 28, 42, 0.75);
                min-height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .forecast-label {
                font-size: 0.82rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: rgba(237,246,255,0.85);
                margin-bottom: 0.6rem;
            }
            .forecast-value {
                font-size: 3rem;
                font-weight: 800;
                line-height: 1;
                color: white;
            }
            .forecast-category {
                font-size: 0.92rem;
                font-weight: 700;
                margin-top: 0.5rem;
            }
            .page-card {
                background: rgba(12, 27, 41, 0.8);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 14px 30px rgba(0,0,0,0.18);
                margin-bottom: 1rem;
            }
            .stTable table {
                border-collapse: collapse;
                background: rgba(5, 15, 25, 0.6);
                color: #edf6ff;
            }
            .stTable th {
                background: rgba(17, 80, 108, 0.85);
                color: white;
            }
            .stAlert {
                border-radius: 14px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.18);
            }
            h1, h2, h3 {
                color: #f0f7ff;
            }
            .stMarkdown p, .stMarkdown li {
                color: #dfeefe;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def load_data():
    return config.load_features()


@st.cache_resource
def load_models():
    models = {}
    for h in config.HORIZONS_DAYS:
        path = config.MODEL_REGISTRY_DIR / f"model_{h}d.joblib"
        if path.exists():
            models[h] = joblib.load(path)
    return models


def latest_forecast(df: pd.DataFrame, models: dict):
    """Use the most recent complete feature row to predict AQI for each horizon."""
    if not models:
        return None, {}
    feature_cols = models[list(models.keys())[0]]["feature_cols"]
    usable = df.dropna(subset=feature_cols)
    if usable.empty:
        return None, {}
    latest_row = usable.iloc[[-1]]
    preds = {}
    for h, bundle in models.items():
        X = latest_row[bundle["feature_cols"]]
        X_s = bundle["scaler"].transform(X)
        preds[h] = float(bundle["model"].predict(X_s)[0])
    return latest_row["time"].iloc[0], preds


def render_summary_cards(as_of, preds):
    if not preds:
        return

    max_pred = max(preds.values())
    latest_value = round(float(max_pred), 0)
    risk_text = "High risk" if latest_value >= 151 else "Elevated" if latest_value >= 100 else "Moderate"

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Peak AQI</div>
                <div class="summary-value">{latest_value}</div>
                <div class="summary-subtext">max predicted value</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Trend</div>
                <div class="summary-value">{ 'Rising' if latest_value > 80 else 'Stable' }</div>
                <div class="summary-subtext">next 3-day pattern</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Risk</div>
                <div class="summary-value">{risk_text}</div>
                <div class="summary-subtext">air quality category</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">As of</div>
                <div class="summary-value" style="font-size: 1.2rem;">{as_of.strftime('%Y-%m-%d %H:%M') if hasattr(as_of, 'strftime') else as_of}</div>
                <div class="summary-subtext">latest forecast snapshot</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_forecast_cards(preds: dict):
    if not preds:
        return
    cols = st.columns(len(preds))
    hazardous_alerts = []
    for col, (h, value) in zip(cols, sorted(preds.items())):
        label, color = config.aqi_category(value)
        with col:
            st.markdown(
                f"""
                <div class="forecast-card" style="background: linear-gradient(180deg, {color}22 0%, rgba(9, 17, 28, 0.82) 100%); border: 1px solid {color};">
                    <div class="forecast-label">Day +{h}</div>
                    <div class="forecast-value">{value:.0f}</div>
                    <div class="forecast-category" style="color: {color};">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if value >= 151:
            hazardous_alerts.append((h, value, label))

    if hazardous_alerts:
        for h, value, label in hazardous_alerts:
            st.warning(
                f"⚠️ **Alert:** AQI is predicted to reach **{value:.0f} ({label})** in {h} day(s) — sensitive groups should limit outdoor exposure."
            )
    else:
        st.success("No hazardous AQI levels predicted in the next 3 days.")
