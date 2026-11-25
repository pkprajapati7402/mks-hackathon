"""
Demand Explorer View (Required View 3).
4-level drill-down (Product Family -> SKU -> Region -> Customer), seasonality heatmap,
zero-demand month analysis, Syntetos-Boylan intermittency quadrants, and ordering-cycle dynamics (Rule 6).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.config import REPORTS_DIR
from src.dashboard.components import render_kpi_card, render_chart, render_table


def render_view(train_df: pd.DataFrame, sku_master: pd.DataFrame):
    st.markdown("### 🔍 Demand Explorer & Granular Profile Analysis")
    st.markdown("Multi-dimensional drill-down (`Family → SKU → Region → Customer`), seasonality decomposition, zero-demand months, and customer cadence dynamics.")

    if train_df.empty:
        st.warning("No matching historical demand for the selected sidebar filter.")
        return

    # 1. Interactive 4-Level Hierarchy Drill-Down (Family -> SKU -> Region -> Customer)
    st.markdown("#### 🌳 Interactive Hierarchy Drill-Down (`Family → Top SKUs → Region → Customer`)")
    top_sku_ids = train_df.groupby("SKU_ID")["Units_Sold_Unconstrained"].sum().nlargest(40).index
    drill_src = train_df[train_df["SKU_ID"].isin(top_sku_ids)]
    drill_cols = [c for c in ["Product_Family", "SKU_ID", "Region", "Customer_Name"] if c in drill_src.columns]
    hierarchy_agg = drill_src.groupby(drill_cols)["Units_Sold_Unconstrained"].sum().reset_index()
    hierarchy_agg = hierarchy_agg[hierarchy_agg["Units_Sold_Unconstrained"] > 0]

    fig_sun = px.sunburst(
        hierarchy_agg,
        path=drill_cols,
        values="Units_Sold_Unconstrained",
        color="Units_Sold_Unconstrained",
        color_continuous_scale="Teal",
        title="Click Any Sector to Drill Down: Family → SKU → Region → Customer (Top 40 Volume SKUs in View)",
    )
    fig_sun.update_layout(template="plotly_dark", paper_bgcolor="#1E293B", height=540)
    render_chart(fig_sun)

    # 2. Seasonality Heatmap & Syntetos-Boylan Quadrants
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📅 Monthly Seasonality Heatmap")
        season_matrix = train_df.groupby(["Year", "Month"])["Units_Sold_Unconstrained"].sum().unstack()
        fig_heat = px.imshow(
            season_matrix,
            labels=dict(x="Month (1-12)", y="Year", color="Units"),
            color_continuous_scale="Viridis",
            title="Portfolio Demand by Month & Year",
        )
        fig_heat.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
        render_chart(fig_heat)

    with col_b:
        st.markdown("#### 🎯 Syntetos-Boylan Intermittency Quadrants")
        interm_series = train_df.groupby(["SKU_ID", "Region"]).agg(
            total_periods=("Date", "count"),
            nonzero_periods=("Units_Sold", lambda s: (s > 0).sum()),
            mean_demand=("Units_Sold", lambda s: s[s > 0].mean() if (s > 0).sum() > 0 else 0),
            std_demand=("Units_Sold", lambda s: s[s > 0].std() if (s > 0).sum() > 1 else 0),
        ).reset_index()

        interm_series["ADI"] = np.where(interm_series["nonzero_periods"] > 0, interm_series["total_periods"] / interm_series["nonzero_periods"], 99.0)
        interm_series["CV2"] = np.where(interm_series["mean_demand"] > 0, (interm_series["std_demand"] / interm_series["mean_demand"]) ** 2, 0.0)
        interm_series["Zero_Months_Pct"] = ((1.0 - interm_series["nonzero_periods"] / np.maximum(interm_series["total_periods"], 1)) * 100.0).round(1)

        def get_cat(row):
            if row["ADI"] < 1.32 and row["CV2"] < 0.49:
                return "Smooth"
            if row["ADI"] >= 1.32 and row["CV2"] < 0.49:
                return "Intermittent"
            if row["ADI"] < 1.32 and row["CV2"] >= 0.49:
                return "Erratic"
            return "Lumpy"

        interm_series["Category"] = interm_series.apply(get_cat, axis=1)

        fig_quad = px.scatter(
            interm_series.sample(min(800, len(interm_series)), random_state=42),
            x="ADI", y="CV2", color="Category",
            title="Demand Classification: Smooth, Intermittent, Erratic, Lumpy",
            range_x=[0.8, 4.0], range_y=[0.0, 2.0],
            template="plotly_dark",
            color_discrete_map={"Smooth": "#10B981", "Intermittent": "#00E5FF", "Erratic": "#F59E0B", "Lumpy": "#EF4444"},
        )
        fig_quad.add_vline(x=1.32, line_dash="dash", line_color="#94A3B8")
        fig_quad.add_hline(y=0.49, line_dash="dash", line_color="#94A3B8")
        fig_quad.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A")
        render_chart(fig_quad)

    # 3. Zero-Demand Months & Ordering Cycle Shifts (Rule 3 & Rule 6)
    st.markdown("#### 🔄 Zero-Month Sparsity, Top SKUs & Customer Ordering-Cycle Dynamics")
    tab_top, tab_zero, tab_cad = st.tabs(["🏆 Top Demand SKUs & Lifecycle", "🕳️ Zero-Demand Months Profile", "📅 Customer Ordering-Cycle Shifts (Rule 6)"])

    with tab_top:
        top_skus = train_df.groupby(["SKU_ID", "Product_Family", "Business_Unit"])["Units_Sold_Unconstrained"].sum().reset_index()
        top_skus = top_skus.sort_values("Units_Sold_Unconstrained", ascending=False).head(25)
        sku_status = dict(zip(sku_master["SKU_ID"], sku_master.get("Lifecycle_Status", ["Active"] * len(sku_master))))
        top_skus["Lifecycle_Status"] = top_skus["SKU_ID"].map(sku_status)
        render_table(top_skus)

    with tab_zero:
        fig_zero = px.histogram(
            interm_series, x="Zero_Months_Pct", nbins=20, color="Category",
            title="Distribution of Zero-Order Month Share (%) Across SKU-Region Series",
            template="plotly_dark",
            color_discrete_map={"Smooth": "#10B981", "Intermittent": "#00E5FF", "Erratic": "#F59E0B", "Lumpy": "#EF4444"},
        )
        fig_zero.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", xaxis_title="% Months with Zero Orders")
        render_chart(fig_zero)

    with tab_cad:
        cad_shift_path = REPORTS_DIR / "cadence_changes.csv"
        if cad_shift_path.exists():
            cad_df = pd.read_csv(cad_shift_path)
            st.caption(f"Detected {len(cad_df):,} Customer-SKU-Region relationships whose ordering cadence shifted between 2019–2022 and 2023–2026.")
            render_table(cad_df.head(100))
        else:
            st.info("Ordering cycle shift report not yet generated.")
