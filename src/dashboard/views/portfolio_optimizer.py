"""
ABC-XYZ Portfolio Segmentation & Working Capital Optimizer View (Competitive Differentiator View 14).
Classifies every SKU by Forecast Revenue Value (ABC) x Demand Interval Volatility (XYZ)
and computes Newsvendor Cycle Stock + Optimal Service-Level Safety Buffers in USD.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import OUTPUTS_DIR
from src.dashboard.components import render_kpi_card, render_chart, render_table


def render_view(forecast_df: pd.DataFrame):
    st.markdown("### 💎 ABC-XYZ Portfolio & Working Capital Optimizer")
    st.markdown("Segments the 601-SKU portfolio into a **9-cell ABC (Revenue) × XYZ (Interval Volatility) Matrix** to allocate service levels (`98% / 95% / 90%`) and minimize idle working capital.")

    abc_path = OUTPUTS_DIR / "abc_xyz_inventory.csv"
    if not abc_path.exists():
        st.warning("ABC-XYZ inventory optimization file not found. Run `python run_pipeline.py` first.")
        return

    abc_df = pd.read_csv(abc_path)
    if not forecast_df.empty:
        active_skus = forecast_df["SKU_ID"].unique()
        abc_df = abc_df[abc_df["SKU_ID"].isin(active_skus)].copy()

    if abc_df.empty:
        st.warning("No matching SKUs under current sidebar filters.")
        return

    # 1. Working Capital KPIs
    tot_rev = abc_df["Forecast_Revenue_USD"].sum()
    tot_wc = abc_df["Working_Capital_USD"].sum()
    class_a_cnt = (abc_df["ABC_Class"] == "A").sum()
    az_cnt = (abc_df["ABC_XYZ_Segment"] == "AZ").sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("19-Mo Forecast Revenue", f"${tot_rev / 1e6:,.1f}M USD", f"{len(abc_df)} Active SKUs in View")
    with col2:
        render_kpi_card("Recommended Working Capital", f"${tot_wc / 1e6:,.2f}M USD", "Cycle Stock + Optimal SLA Buffer")
    with col3:
        render_kpi_card("Tier-A Revenue Champions", f"{class_a_cnt} SKUs", "80% Revenue Concentration (98% SLA)")
    with col4:
        render_kpi_card("AZ High-Risk Critical SKUs", f"{az_cnt} SKUs", "High Revenue × High Uncertainty")

    st.markdown("---")

    # 2. 9-Cell ABC-XYZ Matrix Heatmaps (SKU Count & Working Capital)
    col_a, col_b = st.columns(2)
    with col_a:
        sku_matrix = (
            abc_df.groupby(["ABC_Class", "XYZ_Class"])["SKU_ID"]
            .count()
            .unstack(fill_value=0)
            .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        )
        fig_cnt = px.imshow(
            sku_matrix, text_auto=True, color_continuous_scale="Teal",
            title="ABC × XYZ Portfolio SKU Count Matrix",
            labels=dict(x="Volatility Class (X=Stable, Z=Volatile)", y="Revenue Tier (A=Top 80%)", color="SKUs"),
        )
        fig_cnt.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
        render_chart(fig_cnt)

    with col_b:
        wc_matrix = (
            (abc_df.groupby(["ABC_Class", "XYZ_Class"])["Working_Capital_USD"].sum() / 1e6)
            .round(2)
            .unstack(fill_value=0.0)
            .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0.0)
        )
        fig_wc = px.imshow(
            wc_matrix, text_auto=True, color_continuous_scale="YlOrBr",
            title="Recommended Working Capital Allocation ($M USD)",
            labels=dict(x="Volatility Class", y="Revenue Tier", color="$M USD"),
        )
        fig_wc.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
        render_chart(fig_wc)

    # 3. Critical AZ/AY Priority Table
    st.markdown("#### 🎯 Priority SKU Buffer & Service-Level Allocation Table")
    seg_filter = st.selectbox("Filter ABC-XYZ Segment:", ["All"] + sorted(abc_df["ABC_XYZ_Segment"].unique().tolist()))
    disp_df = abc_df if seg_filter == "All" else abc_df[abc_df["ABC_XYZ_Segment"] == seg_filter]
    show_cols = [
        c for c in [
            "SKU_ID", "Product_Family", "Powertrain_Application", "ABC_XYZ_Segment",
            "Target_Service_Level", "P50", "Interval_Spread_Ratio",
            "Forecast_Revenue_USD", "Cycle_Stock_Units", "Recommended_Safety_Stock_Units", "Working_Capital_USD"
        ] if c in disp_df.columns
    ]
    render_table(disp_df[show_cols].head(50))
