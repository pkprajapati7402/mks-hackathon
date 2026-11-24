"""
Stockouts & Lost Sales Register View (Optional View 5).
Investigates historical supply bottlenecks, unfulfilled customer orders, fill rates,
and lost revenue by sales region and product family.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import CLEANED_DATA_DIR
from src.dashboard.components import render_kpi_card

def render_view():
    st.markdown("### ⚠️ Historical Stockouts & Lost Sales Register")
    st.markdown("Quantifying historical supply-chain bottlenecks, component shortages, order fill rates, and unfulfilled customer demand.")

    stockouts_path = CLEANED_DATA_DIR / "inventory_stockouts.csv"
    if not stockouts_path.exists():
        st.warning("Inventory stockouts data not found.")
        return

    stock_df = pd.read_csv(stockouts_path)

    # 1. Headline Stockout KPIs
    tot_events = len(stock_df)
    tot_requested = stock_df["Requested_Units"].sum()
    tot_shipped = stock_df["Shipped_Units"].sum()
    tot_lost = stock_df["Lost_Units"].sum()
    avg_fill_rate = (tot_shipped / max(1, tot_requested)) * 100.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Stockout Months", f"{tot_events:,} Incidents", "Recorded Supply Bottlenecks")
    with col2:
        render_kpi_card("Total Unfulfilled Lost Demand", f"{tot_lost:,.0f} units", "Never Shipped Shortfall")
    with col3:
        render_kpi_card("Average Fulfillment Rate", f"{avg_fill_rate:.1f}%", f"Shipped vs Customer Requests")
    with col4:
        # Estimated lost revenue at $110/unit
        lost_rev = (tot_lost * 110.0) / 1e6
        render_kpi_card("Estimated Lost Revenue", f"${lost_rev:.2f}M USD", "Unrealized Top-Line")

    st.markdown("---")

    # 2. Regional Fulfillment & Shortage Causes
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🗺️ Lost Units by Commercial Sales Region")
        reg_lost = stock_df.groupby("Region")["Lost_Units"].sum().reset_index()
        fig_reg = px.bar(
            reg_lost.sort_values("Lost_Units", ascending=True),
            x="Lost_Units", y="Region", orientation="h",
            color="Lost_Units", template="plotly_dark",
            color_continuous_scale="Reds",
            title="Lost Demand Volume by Region"
        )
        fig_reg.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", coloraxis_showscale=False)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_b:
        st.markdown("#### 🔍 Root-Cause Shortage Breakdown")
        reason_counts = stock_df["Reason"].value_counts().reset_index()
        fig_reason = px.pie(
            reason_counts, names="Reason", values="count",
            hole=0.45, template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Antique
        )
        fig_reason.update_layout(paper_bgcolor="#1E293B")
        st.plotly_chart(fig_reason, use_container_width=True)

    # 3. Searchable Stockouts Register Table
    st.markdown("#### 📋 Detailed Stockout & Allocation Register")
    reg_filter = st.selectbox("Filter by Region:", ["All"] + sorted(stock_df["Region"].unique().tolist()))
    display_df = stock_df if reg_filter == "All" else stock_df[stock_df["Region"] == reg_filter]

    st.dataframe(
        display_df[["Date", "SKU_ID", "Region", "Requested_Units", "Shipped_Units", "Lost_Units", "Fill_Rate_Pct", "Stockout_Days", "Reason"]].head(250),
        use_container_width=True
    )
    st.caption(f"Showing first {min(250, len(display_df))} of {len(display_df):,} stockout records.")
