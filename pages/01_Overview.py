import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_common import apply_theme, latest_forecast, load_data, load_models, render_forecast_cards, render_summary_cards
import config


apply_theme()
st.set_page_config(page_title=f"Overview - {config.CITY_NAME}", page_icon="🌫️", layout="wide")

st.markdown(
    f"""
    <div class="hero-panel">
        <div class="city-pill">Live AQI outlook • {config.CITY_NAME}</div>
        <h1 style="margin: 0.75rem 0 0.3rem; font-size: 2.7rem; line-height: 1.1;">AQI Forecast Dashboard</h1>
        <p style="margin: 0; color: #dfeeff; font-size: 1.04rem; max-width: 900px;">
            Air-quality intelligence for {config.CITY_NAME}, powered by real-time meteorology, engineered environmental indicators, and predictive modeling.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_data()
except FileNotFoundError:
    st.error("No features found yet. Run `python feature_pipeline.py` first.")
    st.stop()

models = load_models()
if not models:
    st.error("No trained models found yet. Run `python training_pipeline.py` first.")
    st.stop()

as_of, preds = latest_forecast(df, models)
render_summary_cards(as_of, preds)

st.subheader(f"3-Day Forecast (as of {as_of})")
render_forecast_cards(preds)

st.divider()
st.subheader("Recent AQI Trend")
recent = df.dropna(subset=["us_aqi"]).tail(24 * 14)
fig = go.Figure()
fig.add_trace(go.Scatter(x=recent["time"], y=recent["us_aqi"], mode="lines", name="US AQI"))
for lo, hi, label, color in config.AQI_CATEGORIES:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.08, line_width=0)
fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="US AQI")
st.plotly_chart(fig, width="stretch")
