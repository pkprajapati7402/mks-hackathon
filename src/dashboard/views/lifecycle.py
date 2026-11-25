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
from src.dashboard.components import render_kpi_card, render_chart, render_table

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
        status_counts = sku_master["Lifecycle_Status"].value_counts().rename_axis("Lifecycle_Status").reset_index(name="count")
        fig_stat = px.pie(
            status_counts, names="Lifecycle_Status", values="count",
            hole=0.45, template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_stat.update_layout(paper_bgcolor="#1E293B")
        render_chart(fig_stat)

    with col_b:
        st.markdown("#### ⚡ Powertrain Application Distribution")
        pt_counts = sku_master["Powertrain_Application"].value_counts().rename_axis("Powertrain_Application").reset_index(name="count")
        fig_pt = px.bar(
            pt_counts, x="Powertrain_Application", y="count",
            color="Powertrain_Application", template="plotly_dark",
            color_discrete_map={"ICE": "#F59E0B", "EV": "#00E5FF", "Universal": "#10B981"}
        )
        fig_pt.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", coloraxis_showscale=False)
        render_chart(fig_pt)

    # 5. Visual Predecessor-to-Successor Transition Trajectory (2025-2028)
    st.markdown("#### 📈 Predecessor ➔ Successor Transition Ramp Verification (Planned Launches & EOL)")
    pairs = [("PTC-005", "PTC-029"), ("PMP-003", "PMP-020"), ("THM-001", "THM-012")]
    selected_pair = st.selectbox(
        "Inspect Planned EOL ➔ Replacement Transition Pair:",
        [f"{old} (EOL) ➔ {new} (Planned Launch)" for old, new in pairs],
    )
    old_sku, new_sku = selected_pair.split(" (EOL) ➔ ")[0], selected_pair.split(" (EOL) ➔ ")[1].split(" (")[0]
    hist_p = train_df[train_df["SKU_ID"].isin([old_sku, new_sku]) & (train_df["Date"] >= "2025-01")].groupby(["Date", "SKU_ID"])["Units_Sold_Unconstrained"].sum().reset_index().rename(columns={"Units_Sold_Unconstrained": "Units"})
    fcst_p = forecast_df[forecast_df["SKU_ID"].isin([old_sku, new_sku])].groupby(["Date", "SKU_ID"])["Units_Sold_Forecast"].sum().reset_index().rename(columns={"Units_Sold_Forecast": "Units"})
    comb_p = pd.concat([hist_p, fcst_p], ignore_index=True).sort_values(["Date", "SKU_ID"])
    if not comb_p.empty:
        fig_tr = px.line(
            comb_p, x="Date", y="Units", color="SKU_ID", markers=True,
            title=f"Lifecycle Handoff: {old_sku} Phase-Out vs {new_sku} Cold-Start Ramp-Up",
            template="plotly_dark",
            color_discrete_map={old_sku: "#F59E0B", new_sku: "#00E5FF"},
        )
        fig_tr.add_shape(type="line", x0="2026-10", x1="2026-10", y0=0, y1=1, yref="paper", line=dict(color="#94A3B8", width=2, dash="dash"))
        fig_tr.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", yaxis_title="Monthly Units")
        render_chart(fig_tr)

    # 3. Successor Mapping Table (Replaces_SKU)
    st.markdown("#### 🔗 Predecessor-Successor Transition Register")
    replacements = sku_master[sku_master["Replaces_SKU"].notna() & (sku_master["Replaces_SKU"].str.strip() != "")][
        ["SKU_ID", "Product_Family", "SKU_Description", "Launch_Date", "Replaces_SKU", "Lifecycle_Status"]
    ].copy()
    replacements.rename(columns={"SKU_ID": "New_Successor_SKU", "Replaces_SKU": "Predecessor_SKU"}, inplace=True)
    render_table(replacements)

    # 4. Imminent EOL Cutoffs
    st.markdown("#### ⏳ SKUs Approaching End-of-Life (EOL Cutoff Enforcement)")
    eol_skus = sku_master[sku_master["EOL_Date"].notna()][
        ["SKU_ID", "Product_Family", "SKU_Description", "Launch_Date", "EOL_Date", "Lifecycle_Status"]
    ].sort_values("EOL_Date")
    render_table(eol_skus)
