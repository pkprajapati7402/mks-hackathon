"""
What-If Macroeconomic Scenario Simulator View (Optional View 2).
Allows planners to test macroeconomic shocks (e.g. PMI -3, raw material spike, EV surge)
and dynamically projects revised portfolio demand curves in real time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.dashboard.components import render_kpi_card, render_chart

def render_view(forecast_df: pd.DataFrame):
    st.markdown("### 🎛️ What-If Macroeconomic Scenario Simulator")
    st.markdown("Simulate global economic shocks and quantify their impact on Celeris Automobili's forward demand trajectory.")

    # 1. Interactive Shock Controls
    st.markdown("#### ⚙️ Scenario Stress Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        pmi_shock = st.slider("ISM Manufacturing PMI Adjustment:", min_value=-6.0, max_value=+6.0, value=-3.0, step=0.5,
                              help="e.g. PMI -3 simulates industrial contraction.")
    with col2:
        raw_mat_shock = st.slider("Raw Material Cost Index Shock (%):", min_value=-25.0, max_value=+25.0, value=+10.0, step=5.0,
                                  help="e.g. +10% steel/aluminum cost surge.")
    with col3:
        ev_shock = st.slider("EV Adoption Index Acceleration (%):", min_value=-20.0, max_value=+30.0, value=+15.0, step=5.0,
                             help="e.g. +15% accelerated electric vehicle transition.")

    # 2. Dynamic Scenario Demand Recalculation
    # Elasticity coefficients derived from regression importances:
    # PMI: ~1.8% demand impact per PMI point
    # Raw materials: -0.25% demand elasticity
    # EV shock: +0.65% impact on EV components, -0.4% on ICE components
    baseline_tot = forecast_df["Units_Sold_Forecast"].sum()

    # Compute SKU-level shocked adjustments
    pmi_mult = 1.0 + (pmi_shock * 0.018)
    mat_mult = 1.0 - (raw_mat_shock * 0.0025)

    shocked_fcst = forecast_df["Units_Sold_Forecast"].copy().astype(float)
    shocked_fcst = shocked_fcst * pmi_mult * mat_mult

    # Powertrain differential shock
    if "Powertrain_Application" in forecast_df.columns:
        is_ev = forecast_df["Powertrain_Application"] == "EV"
        is_ice = forecast_df["Powertrain_Application"] == "ICE"
        shocked_fcst[is_ev] = shocked_fcst[is_ev] * (1.0 + ev_shock * 0.012)
        shocked_fcst[is_ice] = shocked_fcst[is_ice] * (1.0 - ev_shock * 0.006)

    shocked_tot = shocked_fcst.sum()
    delta_units = shocked_tot - baseline_tot
    delta_pct = (delta_units / max(1, baseline_tot)) * 100.0

    # 3. Scenario Impact KPIs
    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_kpi_card("Baseline 19-Mo Demand", f"{baseline_tot:,.0f} units", "Original Operational Plan")
    with kpi2:
        render_kpi_card("Shocked Scenario Demand", f"{shocked_tot:,.0f} units", f"Simulated Portfolio Volume")
    with kpi3:
        render_kpi_card("Volume Impact (Delta)", f"{delta_units:+,.0f} units", f"{delta_pct:+.2f}% vs Baseline")
    with kpi4:
        # Estimated revenue impact assuming median price $120
        delta_rev = (delta_units * 120.0) / 1e6
        render_kpi_card("Estimated Revenue Delta", f"${delta_rev:+.2f}M USD", "Assuming $120 median unit price")

    # 4. Comparison Timeline Plot
    st.markdown("#### 📈 Baseline Plan vs Simulated Shock Trajectory")
    monthly_base = forecast_df.groupby("Date")["Units_Sold_Forecast"].sum().reset_index()
    monthly_shock = forecast_df.assign(Shocked=shocked_fcst).groupby("Date")["Shocked"].sum().reset_index()

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=monthly_base["Date"],
        y=monthly_base["Units_Sold_Forecast"],
        mode="lines+markers",
        name="Baseline Plan (P50)",
        line=dict(color="#00E5FF", width=2.5)
    ))
    fig_sim.add_trace(go.Scatter(
        x=monthly_shock["Date"],
        y=monthly_shock["Shocked"],
        mode="lines+markers",
        name=f"Simulated Scenario (PMI {pmi_shock:+0.1f}, RawMat {raw_mat_shock:+0.0f}%, EV {ev_shock:+0.0f}%)",
        line=dict(color="#EF4444" if delta_units < 0 else "#10B981", width=2.5, dash="dash")
    ))

    fig_sim.update_layout(
        title="19-Month Forecast Trajectory under Macroeconomic Stress Scenario",
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        yaxis_title="Monthly Units Demand"
    )
    render_chart(fig_sim)
