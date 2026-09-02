import json

import pandas as pd
import streamlit as st

from dashboard_common import apply_theme
import config


apply_theme()
st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")

st.markdown(
    """
    <div class="hero-panel">
        <div class="city-pill">Model Insights</div>
        <h1 style="margin: 0.75rem 0 0.3rem; font-size: 2.4rem; line-height: 1.1;">Performance & Explainability</h1>
        <p style="margin: 0; color: #dfeeff; font-size: 1.02rem; max-width: 900px;">
            Accuracy metrics, model selection, and SHAP-based feature importance for the forecasting pipeline used in this project.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics_path = config.METRICS_PATH
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    rows = []
    for h_key, info in metrics.items():
        best = info["results"][info["best_model"]]
        rows.append({
            "Horizon": h_key,
            "Best model": info["best_model"],
            "RMSE": round(best["rmse"], 2),
            "MAE": round(best["mae"], 2),
            "R²": round(best["r2"], 3),
        })
    st.subheader("Model performance")
    st.table(pd.DataFrame(rows))
else:
    st.info("No metrics file found yet. Run training_pipeline.py first.")

shap_path = config.MODEL_REGISTRY_DIR / "shap_summary_1d.png"
if shap_path.exists():
    st.divider()
    st.subheader("SHAP feature importance (1-day model)")
    st.image(str(shap_path), width="stretch")
else:
    st.info("Run training_pipeline.py to generate the SHAP explainability plot.")
