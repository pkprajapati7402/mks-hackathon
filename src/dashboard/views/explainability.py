"""
AI Explainability & Driver Attribution View (Competitive Differentiator View 13).
Exposes exact native LightGBM Tree-SHAP feature attributions (pred_contrib=True) grouped into
actionable Supply Chain Business Drivers at Portfolio, Family, and SKU-Region granularity.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import OUTPUTS_DIR, REPORTS_DIR
from src.dashboard.components import render_kpi_card, render_chart, render_table

DRIVER_COLS = [
    "Baseline & Seasonal Level",
    "Macro Economy",
    "EV Transition",
    "Calendar & Price Hikes",
    "Customer & Scoped Events",
    "Catalog & Lifecycle",
]


def render_view(forecast_df: pd.DataFrame):
    st.markdown("### 🧠 AI Explainability & Forecast Driver Attribution (Tree-SHAP)")
    st.markdown("Exact LightGBM leaf-path attributions (`pred_contrib=True`) decomposing **WHY** the AI predicts each SKU-Region forecast across 6 business driver groups.")

    drv_path = OUTPUTS_DIR / "forecast_drivers.parquet"
    if not drv_path.exists():
        st.warning("Tree-SHAP forecast drivers file not found. Run `python run_pipeline.py` to generate `outputs/forecast_drivers.parquet`.")
        return

    drv_df = pd.read_parquet(drv_path)
    # Sync with active global sidebar filters
    if not forecast_df.empty:
        active_keys = forecast_df[["Date", "SKU_ID", "Region"]].drop_duplicates()
        drv_df = drv_df.merge(active_keys, on=["Date", "SKU_ID", "Region"], how="inner")

    if drv_df.empty:
        st.warning("No matching records under the active sidebar filters.")
        return

    # 1. Global Attribution KPIs
    avail_drivers = [c for c in DRIVER_COLS if c in drv_df.columns]
    abs_totals = {c: float(drv_df[c].abs().sum()) for c in avail_drivers}
    signed_totals = {c: float(drv_df[c].sum()) for c in avail_drivers}
    tot_abs = max(1.0, sum(abs_totals.values()))

    top_drv = max(abs_totals, key=abs_totals.get)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Dominant Forecast Driver", top_drv, f"{abs_totals[top_drv] / tot_abs * 100:.1f}% of Total Magnitude")
    with col2:
        ev_net = signed_totals.get("EV Transition", 0.0)
        render_kpi_card("EV Shift Net Impact", f"{ev_net:+,.0f} units", "Net Portfolio Powertrain Effect")
    with col3:
        evt_net = signed_totals.get("Customer & Scoped Events", 0.0)
        render_kpi_card("Scoped Events Net Impact", f"{evt_net:+,.0f} units", "Orion Retool, Helios Expansion, Cadence")
    with col4:
        mac_net = signed_totals.get("Macro Economy", 0.0)
        render_kpi_card("Macro Drivers Net Impact", f"{mac_net:+,.0f} units", "PMI, Orders, Auto YoY, Raw Materials")

    st.markdown("---")

    # 2. Portfolio Driver Importance Bar & Monthly Attribution Stacked Bar
    col_a, col_b = st.columns(2)
    with col_a:
        share_df = pd.DataFrame([
            {"Driver_Group": k, "Absolute_Impact_Units": round(v, 0), "Net_Signed_Units": round(signed_totals[k], 0), "Share_Pct": round(v / tot_abs * 100, 1)}
            for k, v in abs_totals.items()
        ]).sort_values("Absolute_Impact_Units", ascending=True)

        fig_bar = px.bar(
            share_df, x="Absolute_Impact_Units", y="Driver_Group", orientation="h",
            color="Net_Signed_Units", color_continuous_scale="RdYlGn",
            title="Portfolio Driver Attribution Magnitude & Net Sign (Units)",
            text="Share_Pct", template="plotly_dark",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A")
        render_chart(fig_bar)

    with col_b:
        monthly_drv = drv_df.groupby("Date")[avail_drivers].sum().reset_index()
        fig_m = px.bar(
            monthly_drv, x="Date", y=avail_drivers, barmode="relative",
            title="Monthly Forecast Driver Decomposition (Nov 2026 – May 2028)",
            template="plotly_dark",
        )
        fig_m.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", yaxis_title="Attributed Units")
        render_chart(fig_m)

    # 3. Interactive Single SKU-Region Waterfall Explainer
    st.markdown("#### 🔍 Interactive SKU-Region Driver Waterfall Explainer")
    col_s, col_r = st.columns(2)
    sku_choices = sorted(drv_df["SKU_ID"].unique().tolist())
    with col_s:
        sel_sku = st.selectbox("Select SKU to Explain:", sku_choices, index=0)
    reg_choices = sorted(drv_df[drv_df["SKU_ID"] == sel_sku]["Region"].unique().tolist())
    with col_r:
        sel_reg = st.selectbox("Select Region:", reg_choices, index=0)

    sub_sku = drv_df[(drv_df["SKU_ID"] == sel_sku) & (drv_df["Region"] == sel_reg)].sort_values("Date")
    if not sub_sku.empty:
        totals = sub_sku[avail_drivers].sum()
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(avail_drivers) + ["total"],
            x=avail_drivers + ["Total 19-Mo Forecast"],
            y=list(totals.values) + [float(sub_sku["P50_Forecast"].sum())],
            increasing=dict(marker=dict(color="#10B981")),
            decreasing=dict(marker=dict(color="#EF4444")),
            totals=dict(marker=dict(color="#00E5FF")),
        ))
        fig_wf.update_layout(
            title=f"SHAP Driver Waterfall for {sel_sku} ({sel_reg}) — 19-Month Total",
            template="plotly_dark", paper_bgcolor="#1E293B", plot_bgcolor="#0F172A",
            yaxis_title="Units Contribution",
        )
        render_chart(fig_wf)

        fi_path = REPORTS_DIR / "feature_importances.csv"
        if fi_path.exists():
            with st.expander("View Top Granular LightGBM Features"):
                render_table(pd.read_csv(fi_path).head(20))
