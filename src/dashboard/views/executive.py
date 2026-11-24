"""
Executive Overview View (Required View 1).
Continuous timeline of history and 19-month forecast, portfolio KPI summary,
and volume mix breakdown by Product Family, Region, and Channel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard.components import render_kpi_card, plot_continuous_timeline, render_chart

def render_view(train_df: pd.DataFrame, forecast_df: pd.DataFrame, test_raw: pd.DataFrame):
    st.markdown("### 🏛️ Executive Overview & Continuous Demand Horizon")
    st.markdown("Seamless transition from historical unconstrained demand (2019-01 to 2026-10) into the 19-month forward forecast (2026-11 to 2028-05).")

    if train_df.empty or forecast_df.empty:
        st.warning("No matching demand records for the selected sidebar filter combination.")
        return

    # 1. Headline KPIs
    tot_fcst = forecast_df["Units_Sold_Forecast"].sum() if "Units_Sold_Forecast" in forecast_df.columns else forecast_df["P50"].sum()
    p10_sum = forecast_df["P10"].sum() if "P10" in forecast_df.columns else tot_fcst * 0.85
    p90_sum = forecast_df["P90"].sum() if "P90" in forecast_df.columns else tot_fcst * 1.15
    active_skus = forecast_df["SKU_ID"].nunique()
    active_regions = forecast_df["Region"].nunique()
    hist_units = train_df["Units_Sold_Unconstrained"].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("19-Month Forecast (P50)", f"{tot_fcst:,.0f} units", "Baseline Operational Plan")
    with col2:
        render_kpi_card("Uncertainty Range [P10-P90]", f"{p10_sum:,.0f} – {p90_sum:,.0f}", f"Spread: ±{(p90_sum-p10_sum)/(2*max(1,tot_fcst))*100:.1f}%")
    with col3:
        render_kpi_card("Active Portfolio Scope", f"{active_skus} SKUs", f"{active_regions} Global Commercial Regions")
    with col4:
        render_kpi_card("Historical Unconstrained Volume", f"{hist_units:,.0f} units", "2019-01 to 2026-10 (Cleaned)")

    st.markdown("---")

    # 2. Continuous Timeline Chart
    # Aggregate monthly history and forecast
    hist_monthly = train_df.groupby("Date").agg({
        "Units_Sold": "sum",
        "Units_Sold_Unconstrained": "sum"
    }).reset_index()

    fcst_monthly = forecast_df.groupby("Date").agg({
        "P10": "sum",
        "P50": "sum",
        "P90": "sum",
        "Units_Sold_Forecast": "sum"
    }).reset_index()

    fig_timeline = plot_continuous_timeline(
        hist_monthly,
        fcst_monthly,
        title="Global Portfolio Units: Historical Actuals, Unconstrained Demand & 19-Month Forecast Intervals"
    )
    render_chart(fig_timeline)

    # 3. Portfolio Mix Breakdowns
    st.markdown("#### 📊 Forward Demand Mix Breakdown (Nov 2026 – May 2028)")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        fam_mix = forecast_df.groupby("Product_Family")["Units_Sold_Forecast"].sum().reset_index()
        fig_fam = px.pie(
            fam_mix, names="Product_Family", values="Units_Sold_Forecast",
            title="Mix by Product Family",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_fam.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
        render_chart(fig_fam)

    with col_b:
        reg_mix = forecast_df.groupby("Region")["Units_Sold_Forecast"].sum().reset_index()
        fig_reg = px.bar(
            reg_mix.sort_values("Units_Sold_Forecast", ascending=True),
            x="Units_Sold_Forecast", y="Region", orientation="h",
            title="Volume by Region",
            color="Units_Sold_Forecast",
            color_continuous_scale="Teal"
        )
        fig_reg.update_layout(template="plotly_dark", paper_bgcolor="#1E293B", coloraxis_showscale=False)
        render_chart(fig_reg)

    with col_c:
        chan_mix = test_raw.merge(forecast_df[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]], on=["Date", "SKU_ID", "Region"])
        chan_agg = chan_mix.groupby("Sales_Channel")["Units_Sold_Forecast"].sum().reset_index()
        fig_chan = px.pie(
            chan_agg, names="Sales_Channel", values="Units_Sold_Forecast",
            title="Mix by Sales Channel",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_chan.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
        render_chart(fig_chan)
