"""
Executive Overview View (Required View 1).
Displays Units AND Revenue continuous timelines (2019-01 -> 2028-05), mix by Family, Region, and Channel,
and headline 19-month forecast & revenue KPIs.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard.components import render_kpi_card, plot_continuous_timeline, render_chart


def render_view(train_df: pd.DataFrame, forecast_df: pd.DataFrame, test_raw: pd.DataFrame):
    st.markdown("### 🏛️ Executive Overview & Continuous Demand Horizon")
    st.markdown("Seamless transition from historical unconstrained demand (2019-01 to 2026-10) into the 19-month forward forecast (2026-11 to 2028-05) across both **Units** and **Revenue ($ USD)**.")

    if train_df.empty or forecast_df.empty:
        st.warning("No matching demand records for the selected sidebar filter combination.")
        return

    # 1. Headline KPIs (Units + Revenue + Intervals)
    tot_fcst = forecast_df["Units_Sold_Forecast"].sum() if "Units_Sold_Forecast" in forecast_df.columns else forecast_df["P50"].sum()
    p10_sum = forecast_df["P10"].sum() if "P10" in forecast_df.columns else tot_fcst * 0.85
    p90_sum = forecast_df["P90"].sum() if "P90" in forecast_df.columns else tot_fcst * 1.15
    proj_rev = (
        forecast_df["Projected_Revenue_USD"].sum()
        if "Projected_Revenue_USD" in forecast_df.columns
        else tot_fcst * float(train_df["Unit_Price_USD"].median() if "Unit_Price_USD" in train_df.columns else 120.0)
    )
    active_skus = forecast_df["SKU_ID"].nunique()
    active_regions = forecast_df["Region"].nunique()
    hist_units = train_df["Units_Sold_Unconstrained"].sum()
    hist_rev = train_df["Revenue_USD"].sum() if "Revenue_USD" in train_df.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("19-Month Forecast (P50)", f"{tot_fcst:,.0f} units", f"Projected Revenue: ${proj_rev / 1e6:,.1f}M USD")
    with col2:
        render_kpi_card("Conformal Range [P10–P90]", f"{p10_sum / 1e6:,.2f}M – {p90_sum / 1e6:,.2f}M", f"Spread: ±{(p90_sum - p10_sum) / (2 * max(1, tot_fcst)) * 100:.1f}%")
    with col3:
        render_kpi_card("Active Portfolio Scope", f"{active_skus} SKUs", f"{active_regions} Global Commercial Regions")
    with col4:
        render_kpi_card("Historical Unconstrained", f"{hist_units / 1e6:,.2f}M units", f"Realized Revenue: ${hist_rev / 1e9:,.2f}B USD")

    st.markdown("---")

    # 2. Dual Continuous Timelines (Units + Revenue Trend per PDF Sec 5)
    tab_units, tab_rev = st.tabs(["📦 Continuous Units Horizon", "💵 Continuous Revenue ($ USD) Horizon"])

    with tab_units:
        hist_monthly = train_df.groupby("Date").agg({
            "Units_Sold": "sum",
            "Units_Sold_Unconstrained": "sum",
        }).reset_index()

        fcst_monthly = forecast_df.groupby("Date").agg({
            "P10": "sum",
            "P50": "sum",
            "P90": "sum",
            "Units_Sold_Forecast": "sum",
        }).reset_index()

        fig_timeline = plot_continuous_timeline(
            hist_monthly,
            fcst_monthly,
            title="Portfolio Units: Historical Shipped, Unconstrained Demand & Calibrated P10–P90 Forecast",
        )
        render_chart(fig_timeline)

    with tab_rev:
        if "Revenue_USD" in train_df.columns:
            hist_rev_m = train_df.groupby("Date")["Revenue_USD"].sum().reset_index()
            if "Projected_Revenue_USD" in forecast_df.columns:
                fcst_rev_m = forecast_df.groupby("Date").agg(
                    Revenue_P50=("Projected_Revenue_USD", "sum"),
                    Unit_Price=("Unit_Price_USD", "median"),
                    P10=("P10", "sum"),
                    P90=("P90", "sum"),
                ).reset_index()
                # Scale P10/P90 revenue bounds cleanly by average forecast price
                avg_px = (fcst_rev_m["Revenue_P50"] / forecast_df.groupby("Date")["Units_Sold_Forecast"].sum().clip(lower=1).values)
                fcst_rev_m["Revenue_P10"] = fcst_rev_m["P10"] * avg_px
                fcst_rev_m["Revenue_P90"] = fcst_rev_m["P90"] * avg_px
            else:
                med_px = float(train_df["Unit_Price_USD"].median()) if "Unit_Price_USD" in train_df.columns else 120.0
                fcst_rev_m = forecast_df.groupby("Date").agg(
                    Revenue_P50=("Units_Sold_Forecast", lambda s: s.sum() * med_px),
                    Revenue_P10=("P10", lambda s: s.sum() * med_px),
                    Revenue_P90=("P90", lambda s: s.sum() * med_px),
                ).reset_index()

            fig_rev = go.Figure()
            fig_rev.add_trace(go.Scatter(
                x=hist_rev_m["Date"], y=hist_rev_m["Revenue_USD"],
                mode="lines", name="Historical Revenue (USD)", line=dict(color="#00E5FF", width=2.4)
            ))
            fig_rev.add_trace(go.Scatter(
                x=fcst_rev_m["Date"], y=fcst_rev_m["Revenue_P90"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig_rev.add_trace(go.Scatter(
                x=fcst_rev_m["Date"], y=fcst_rev_m["Revenue_P10"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.16)", name="Projected Revenue [P10–P90]"
            ))
            fig_rev.add_trace(go.Scatter(
                x=fcst_rev_m["Date"], y=fcst_rev_m["Revenue_P50"],
                mode="lines+markers", name="Projected Revenue (P50)",
                line=dict(color="#10B981", width=2.8), marker=dict(size=5)
            ))
            fig_rev.add_shape(type="line", x0="2026-10", x1="2026-10", y0=0, y1=1, yref="paper", line=dict(color="#94A3B8", width=2, dash="dash"))
            fig_rev.add_annotation(x="2026-10", y=1.0, yref="paper", text="Forecast Window (Nov 2026)", showarrow=False, xanchor="left", yanchor="top")
            fig_rev.update_layout(
                title="Continuous Revenue Trend: History vs 19-Month Forecast (USD)",
                template="plotly_dark", paper_bgcolor="#1E293B", plot_bgcolor="#0F172A",
                yaxis_title="Monthly Revenue (USD)", margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            render_chart(fig_rev)

    # 3. Portfolio Mix Breakdowns (Family, Region, Channel)
    st.markdown("#### 📊 Forward Demand Mix Breakdown (Nov 2026 – May 2028)")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        fam_mix = forecast_df.groupby("Product_Family")["Units_Sold_Forecast"].sum().reset_index()
        fig_fam = px.pie(
            fam_mix, names="Product_Family", values="Units_Sold_Forecast",
            title="Mix by Product Family", hole=0.45, color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_fam.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
        render_chart(fig_fam)

    with col_b:
        reg_mix = forecast_df.groupby("Region")["Units_Sold_Forecast"].sum().reset_index()
        fig_reg = px.bar(
            reg_mix.sort_values("Units_Sold_Forecast", ascending=True),
            x="Units_Sold_Forecast", y="Region", orientation="h",
            title="Volume by Region", color="Units_Sold_Forecast", color_continuous_scale="Teal"
        )
        fig_reg.update_layout(template="plotly_dark", paper_bgcolor="#1E293B", coloraxis_showscale=False)
        render_chart(fig_reg)

    with col_c:
        if "Sales_Channel" in forecast_df.columns:
            chan_agg = forecast_df.groupby("Sales_Channel")["Units_Sold_Forecast"].sum().reset_index()
        else:
            chan_mix = test_raw.merge(forecast_df[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]], on=["Date", "SKU_ID", "Region"])
            chan_agg = chan_mix.groupby("Sales_Channel")["Units_Sold_Forecast"].sum().reset_index()
        fig_chan = px.pie(
            chan_agg, names="Sales_Channel", values="Units_Sold_Forecast",
            title="Mix by Sales Channel", hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_chan.update_layout(template="plotly_dark", paper_bgcolor="#1E293B")
        render_chart(fig_chan)
