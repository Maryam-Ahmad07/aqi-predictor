import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_common import apply_theme, load_data
import config


apply_theme()
st.set_page_config(page_title="Analysis", page_icon="📊", layout="wide")

st.markdown(
    """
    <div class="hero-panel">
        <div class="city-pill">Analysis</div>
        <h1 style="margin: 0.75rem 0 0.3rem; font-size: 2.4rem; line-height: 1.1;">Environmental Signals</h1>
        <p style="margin: 0; color: #dfeeff; font-size: 1.02rem; max-width: 900px;">
            Pollutant behavior, diurnal patterns, and the dominant conditions driving AQI changes for this project.
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

recent = df.dropna(subset=["us_aqi"]).tail(24 * 14)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Pollutant levels (last 14 days, hourly)**")
    pollutant_fig = go.Figure()
    for p in ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"]:
        pollutant_fig.add_trace(go.Scatter(x=recent["time"], y=recent[p], mode="lines", name=p))
    pollutant_fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(pollutant_fig, width="stretch")
with c2:
    st.markdown("**AQI by hour of day (avg, all history)**")
    by_hour = df.dropna(subset=["us_aqi"]).groupby("hour")["us_aqi"].mean().reset_index()
    hour_fig = go.Figure(go.Bar(x=by_hour["hour"], y=by_hour["us_aqi"]))
    hour_fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour", yaxis_title="Avg AQI")
    st.plotly_chart(hour_fig, width="stretch")

st.divider()
st.subheader("AQI category bands")
for lo, hi, label, color in config.AQI_CATEGORIES:
    st.markdown(
        f"<div class='page-card' style='border-left: 6px solid {color};'>"
        f"<strong>{label}</strong> — {lo} to {hi}</div>",
        unsafe_allow_html=True,
    )
