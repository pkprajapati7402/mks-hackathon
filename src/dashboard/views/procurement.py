"""
Procurement Plan from P90 Forecasts View (Optional View 3).
Translates P90 demand projections into component procurement schedules,
safety stock sizing (P90 - P50), and inventory risk management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard.components import render_kpi_card

def render_view(forecast_df: pd.DataFrame):
    st.markdown("### 📦 Procurement Schedule & Safety Stock Planning (P90)")
    st.markdown("Translates high-confidence upper quantile demand forecasts (P90) into actionable supplier procurement orders and safety buffer recommendations.")

    # Compute Buffer Metrics
    p50_tot = forecast_df["P50"].sum()
    p90_tot = forecast_df["P90"].sum()
    buffer_tot = p90_tot - p50_tot
    buffer_pct = (buffer_tot / max(1, p50_tot)) * 100.0

    # 1. Headline Procurement KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("P90 Total Procurement", f"{p90_tot:,.0f} units", "Upper Bound Capacity Commitment")
    with col2:
        render_kpi_card("Safety Buffer Volume", f"{buffer_tot:,.0f} units", f"+{buffer_pct:.1f}% above P50 median")
    with col3:
        # Estimated holding cost of buffer at $15/unit/year
        holding_cost = (buffer_tot * 15.0) / 1e6
        render_kpi_card("Estimated Buffer Holding Cost", f"${holding_cost:.2f}M USD", "Annualized Inventory Carrying")
    with col4:
        # Stockout protection SLA
        render_kpi_card("Stockout Protection SLA", "90% Service Level", "Standard Automotive Tier-1 Target")

    st.markdown("---")

    # 2. Monthly Procurement Scheduling Timeline
    st.markdown("#### 📅 Monthly Procurement Plan: Baseline (P50) vs Buffer Order (P90)")
    monthly_proc = forecast_df.groupby("Date").agg({
        "P10": "sum",
        "P50": "sum",
        "P90": "sum"
    }).reset_index()
    monthly_proc["Safety_Buffer"] = monthly_proc["P90"] - monthly_proc["P50"]

    fig_proc = go.Figure()
    fig_proc.add_trace(go.Bar(
        x=monthly_proc["Date"],
        y=monthly_proc["P50"],
        name="Baseline Order Volume (P50)",
        marker_color="#00E5FF"
    ))
    fig_proc.add_trace(go.Bar(
        x=monthly_proc["Date"],
        y=monthly_proc["Safety_Buffer"],
        name="Recommended Safety Buffer (P90 - P50)",
        marker_color="#F59E0B"
    ))

    fig_proc.update_layout(
        barmode="stack",
        title="Monthly Component Procurement Plan (Nov 2026 – May 2028)",
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        yaxis_title="Recommended Purchase Order Units"
    )
    st.plotly_chart(fig_proc, use_container_width=True)

    # 3. High Uncertainty SKUs (Procurement Risk Ranking)
    st.markdown("#### ⚠️ High Uncertainty SKUs Requiring Buffer Allocation Priority")
    sku_proc = forecast_df.groupby(["SKU_ID", "Product_Family", "Region"]).agg({
        "P50": "sum",
        "P90": "sum"
    }).reset_index()
    sku_proc["Safety_Buffer_Units"] = sku_proc["P90"] - sku_proc["P50"]
    sku_proc["Uncertainty_Ratio"] = (sku_proc["P90"] / np.maximum(sku_proc["P50"], 1.0)).round(2)
    
    high_risk_skus = sku_proc[sku_proc["P50"] > 500].sort_values("Uncertainty_Ratio", ascending=False).head(20)
    st.dataframe(high_risk_skus, use_container_width=True)
