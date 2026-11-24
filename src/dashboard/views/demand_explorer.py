"""
Demand Explorer View (Required View 3).
Hierarchical drill-down from Business Unit -> Product Family -> SKU -> Region -> Customer,
seasonality decomposition, zero-demand analysis, and Syntetos-Boylan intermittency profiling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard.components import render_kpi_card

def render_view(train_df: pd.DataFrame, sku_master: pd.DataFrame):
    st.markdown("### 🔍 Demand Explorer & Granular Profile Analysis")
    st.markdown("Interactive multi-dimensional drill-down into historical demand patterns, seasonal profiles, and intermittency classifications.")

    # 1. Hierarchical Sunburst Drilldown
    st.markdown("#### 🌳 Portfolio Hierarchy Decomposition (Units Sold)")
    hierarchy_agg = train_df.groupby(["Business_Unit", "Product_Family", "Region"])["Units_Sold_Unconstrained"].sum().reset_index()
    fig_sun = px.sunburst(
        hierarchy_agg,
        path=["Business_Unit", "Product_Family", "Region"],
        values="Units_Sold_Unconstrained",
        color="Units_Sold_Unconstrained",
        color_continuous_scale="Teal",
        title="Interactive Portfolio Hierarchy: BU ➔ Family ➔ Region"
    )
    fig_sun.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
    st.plotly_chart(fig_sun, use_container_width=True)

    # 2. Seasonality Heatmap & Zero-Demand Streaks
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📅 Monthly Seasonality Heatmap")
        season_matrix = train_df.groupby(["Year", "Month"])["Units_Sold_Unconstrained"].sum().unstack()
        fig_heat = px.imshow(
            season_matrix,
            labels=dict(x="Month (1-12)", y="Year", color="Units"),
            color_continuous_scale="Viridis",
            title="Portfolio Demand by Month & Year"
        )
        fig_heat.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_b:
        st.markdown("#### 🎯 Syntetos-Boylan Intermittency Quadrants")
        # Compute summary distribution of ADI vs CV2
        interm_series = train_df.groupby(["SKU_ID", "Region"]).agg(
            total_periods=("Date", "count"),
            nonzero_periods=("Units_Sold", lambda s: (s > 0).sum()),
            mean_demand=("Units_Sold", lambda s: s[s > 0].mean() if (s > 0).sum() > 0 else 0),
            std_demand=("Units_Sold", lambda s: s[s > 0].std() if (s > 0).sum() > 1 else 0)
        ).reset_index()

        interm_series["ADI"] = np.where(interm_series["nonzero_periods"] > 0, interm_series["total_periods"] / interm_series["nonzero_periods"], 99.0)
        interm_series["CV2"] = np.where(interm_series["mean_demand"] > 0, (interm_series["std_demand"] / interm_series["mean_demand"])**2, 0.0)
        
        # Categorize
        def get_cat(row):
            if row["ADI"] < 1.32 and row["CV2"] < 0.49: return "Smooth"
            if row["ADI"] >= 1.32 and row["CV2"] < 0.49: return "Intermittent"
            if row["ADI"] < 1.32 and row["CV2"] >= 0.49: return "Erratic"
            return "Lumpy"
            
        interm_series["Category"] = interm_series.apply(get_cat, axis=1)

        fig_quad = px.scatter(
            interm_series.sample(min(800, len(interm_series)), random_state=42),
            x="ADI", y="CV2", color="Category",
            title="Demand Classification: Smooth, Intermittent, Erratic, Lumpy",
            range_x=[0.8, 4.0], range_y=[0.0, 2.0],
            template="plotly_dark",
            color_discrete_map={
                "Smooth": "#10B981",
                "Intermittent": "#00E5FF",
                "Erratic": "#F59E0B",
                "Lumpy": "#EF4444"
            }
        )
        fig_quad.add_vline(x=1.32, line_dash="dash", line_color="#94A3B8")
        fig_quad.add_hline(y=0.49, line_dash="dash", line_color="#94A3B8")
        fig_quad.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A")
        st.plotly_chart(fig_quad, use_container_width=True)

    # 3. Top SKUs & Catalog Lifecycles
    st.markdown("#### 🏆 Top Demand Generating SKUs & Active Status")
    top_skus = train_df.groupby(["SKU_ID", "Product_Family", "Business_Unit"])["Units_Sold_Unconstrained"].sum().reset_index()
    top_skus = top_skus.sort_values("Units_Sold_Unconstrained", ascending=False).head(15)
    
    sku_status = dict(zip(sku_master["SKU_ID"], sku_master.get("Lifecycle_Status", ["Active"]*len(sku_master))))
    top_skus["Lifecycle_Status"] = top_skus["SKU_ID"].map(sku_status)
    st.dataframe(top_skus, use_container_width=True)
