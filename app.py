"""AQI dashboard landing page and project navigation."""

import streamlit as st

from dashboard_common import apply_theme
import config


apply_theme()
st.set_page_config(page_title=f"{config.CITY_NAME} AQI Hub", page_icon="🌫️", layout="wide")

st.markdown(
    f"""
    <div class="hero-panel">
        <div class="city-pill">Project Dashboard • {config.CITY_NAME}</div>
        <h1 style="margin: 0.75rem 0 0.3rem; font-size: 2.8rem; line-height: 1.1;">AQI Intelligence Hub</h1>
        <p style="margin: 0; color: #dfeeff; font-size: 1.04rem; max-width: 900px;">
            A professional air-quality forecasting system for {config.CITY_NAME}, covering forecasting, analysis, and model explainability in a multi-page experience.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Navigate the project")

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/01_Overview.py", label="Overview", icon="🏙️")
    st.caption("Forecast cards, summary metrics, and recent AQI trend")
with col2:
    st.page_link("pages/02_Analysis.py", label="Analysis", icon="📊")
    st.caption("Pollutant behavior, hourly AQI patterns, and air-quality bands")
with col3:
    st.page_link("pages/03_Model_Insights.py", label="Model Insights", icon="🧠")
    st.caption("Performance table and SHAP feature-importance image")

st.divider()
st.subheader("Project flow")

st.markdown(
    """
    - Data acquisition: Open-Meteo weather + pollutant inputs
    - Feature engineering: lagged, rolling, and time-based predictors
    - Modeling: multi-horizon AQI regression models
    - Explainability: SHAP-driven feature importance
    - Deployment: a clean dashboard for monitoring and forecasting
    """
)

st.info("Use the sidebar pages to move between the overview, analysis, and model insight sections.")
