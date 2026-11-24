"""
Catalog Lifecycle & Transition Management View (Optional View 4).
Monitors product launches, EOL phase-outs, predecessor-successor SKU handoffs (Replaces_SKU),
and active catalog distributions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard.components import render_kpi_card

def render_view(sku_master: pd.DataFrame, train_df: pd.DataFrame, forecast_df: pd.DataFrame):
    st.markdown("### 🔄 Catalog Lifecycle & Transition Management")
    st.markdown("Tracking component introductions, scheduled End-of-Life (EOL) phase-outs, and predecessor-to-successor transitions.")

    # 1. Headline KPIs
    tot_skus = len(sku_master)
    active_count = (sku_master["Lifecycle_Status"] == "Active").sum()
    launch_count = (sku_master["Lifecycle_Status"] == "Planned Launch").sum()
    eol_count = sku_master["EOL_Date"].notna().sum()
    replaces_count = sku_master["Replaces_SKU"].notna().sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Master Catalog", f"{tot_skus} SKUs", "Covering 7 Product Families")
    with col2:
        render_kpi_card("Planned Launches", f"{launch_count} SKUs", "Commercial Start Post-2026")
    with col3:
        render_kpi_card("Announced EOL SKUs", f"{eol_count} SKUs", "Phased Commercial Retirement")
    with col4:
        render_kpi_card("Successor Handoffs", f"{replaces_count} SKUs", "Mapped via Replaces_SKU")

    st.markdown("---")

    # 2. Lifecycle Status Mix & Powertrain Mix
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📊 SKUs by Lifecycle Status")
        status_counts = sku_master["Lifecycle_Status"].value_counts().reset_index()
        fig_stat = px.pie(
            status_counts, names="Lifecycle_Status", values="count",
            hole=0.45, template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_stat.update_layout(paper_bgcolor="#1E293B")
        st.plotly_chart(fig_stat, use_container_width=True)

    with col_b:
        st.markdown("#### ⚡ Powertrain Application Distribution")
        pt_counts = sku_master["Powertrain_Application"].value_counts().reset_index()
        fig_pt = px.bar(
            pt_counts, x="Powertrain_Application", y="count",
            color="Powertrain_Application", template="plotly_dark",
            color_discrete_map={"ICE": "#F59E0B", "EV": "#00E5FF", "Universal": "#10B981"}
        )
        fig_pt.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", coloraxis_showscale=False)
        st.plotly_chart(fig_pt, use_container_width=True)

    # 3. Successor Mapping Table (Replaces_SKU)
    st.markdown("#### 🔗 Predecessor-Successor Transition Register")
    replacements = sku_master[sku_master["Replaces_SKU"].notna() & (sku_master["Replaces_SKU"].str.strip() != "")][
        ["SKU_ID", "Product_Family", "SKU_Description", "Launch_Date", "Replaces_SKU", "Lifecycle_Status"]
    ].copy()
    replacements.rename(columns={"SKU_ID": "New_Successor_SKU", "Replaces_SKU": "Predecessor_SKU"}, inplace=True)
    st.dataframe(replacements, use_container_width=True)

    # 4. Imminent EOL Cutoffs
    st.markdown("#### ⏳ SKUs Approaching End-of-Life (EOL Cutoff Enforcement)")
    eol_skus = sku_master[sku_master["EOL_Date"].notna()][
        ["SKU_ID", "Product_Family", "SKU_Description", "Launch_Date", "EOL_Date", "Lifecycle_Status"]
    ].sort_values("EOL_Date")
    st.dataframe(eol_skus, use_container_width=True)
