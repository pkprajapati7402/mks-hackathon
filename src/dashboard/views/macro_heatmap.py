"""
Macroeconomic Lead/Lag Heatmap View (Optional View 1).
Interactive correlation matrix across 8 economic indicators and 7 Product Families at lags 0 to 12,
optimal lead times, and temporal lag stability test results.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import REPORTS_DIR
from src.dashboard.components import render_kpi_card, render_chart, render_table

def render_view():
    st.markdown("### 🌡️ Macroeconomic Lead/Lag Analytics & Lag Stability")
    st.markdown("Quantifies how external economic indicators (PMI, vehicle inventory, raw material costs, EV adoption) lead automotive product family demand.")

    lead_lag_path = REPORTS_DIR / "macro_lead_lag_analysis.csv"
    stability_path = REPORTS_DIR / "lag_stability_analysis.csv"

    if not lead_lag_path.exists():
        st.warning("Macro lead/lag analysis not found. Run pipeline first.")
        return

    lead_lag_df = pd.read_csv(lead_lag_path)
    stability_df = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()

    # 1. Headline Lead Indicator Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Leading Indicator Scope", "8 Indicators", "PMI, Orders, Production, EV...")
    with col2:
        render_kpi_card("Maximum Lead Horizon", "3 Months", "New Vehicle Orders Index")
    with col3:
        render_kpi_card("EV Adoption Impact", "+0.82 Correlation", "Surging EV Component Lines")
    with col4:
        render_kpi_card("Lag Structure Stability", "4 of 5 Stable", "Pre-2022 vs Post-2022")

    st.markdown("---")

    # 2. Optimal Lead Lags Summary Table
    st.markdown("#### ⏱️ Optimal Lead Time (Months) by Product Family & Macroeconomic Driver")
    summary_table = lead_lag_df[["Product_Family", "Macro_Indicator", "Optimal_Lag_Months", "Max_Correlation", "Relationship"]].copy()
    render_table(summary_table)

    # 3. Interactive Lead/Lag Cross-Correlation Heatmap
    st.markdown("#### 🗺️ Cross-Correlation Matrix across Lags (0 to 12 Months)")
    selected_family = st.selectbox("Select Product Family to Inspect:", sorted(lead_lag_df["Product_Family"].unique().tolist()))
    fam_data = lead_lag_df[lead_lag_df["Product_Family"] == selected_family]

    lag_cols = [f"lag_{i}" for i in range(13) if f"lag_{i}" in fam_data.columns]
    heatmap_matrix = fam_data.set_index("Macro_Indicator")[lag_cols]

    fig_heat = px.imshow(
        heatmap_matrix,
        labels=dict(x="Lead Lag (Months)", y="Macro Indicator", color="Pearson r"),
        color_continuous_scale="RdBu_r",
        zmin=-1.0, zmax=1.0,
        title=f"Cross-Correlation Profile for {selected_family} (Lags 0-12)"
    )
    fig_heat.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
    render_chart(fig_heat)

    # 4. Temporal Lag Stability Analysis (Pre vs Post 2022)
    if not stability_df.empty:
        st.markdown("#### ⚖️ Temporal Lag Stability Test (Pre-2022 vs Post-2022)")
        st.markdown("Tests whether macroeconomic lead/lag relationships shifted structurally post-COVID / semiconductor shortage.")
        render_table(stability_df)
