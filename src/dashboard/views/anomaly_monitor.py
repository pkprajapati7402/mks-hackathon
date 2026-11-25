"""
Anomaly Monitor View (Required View 4).
Displays interactive timeline scatter plot across all 10 anomaly categories,
searchable Anomaly Register, Customer Switches, and Ordering-Cycle Shifts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.config import REPORTS_DIR
from src.dashboard.components import render_kpi_card, render_chart, render_table


def render_view(train_df: pd.DataFrame):
    st.markdown("### 🚨 Anomaly Monitor & Anomaly Register")
    st.markdown("Separates historical structural breaks, supply stockouts, early-buying pull-forwards, holiday dips, and customer events from ERP errors.")

    reg_path = REPORTS_DIR / "anomaly_register.csv"
    if not reg_path.exists():
        st.warning("Anomaly register not found. Run pipeline first.")
        return

    anom_df = pd.read_csv(reg_path)

    # 1. Headline KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Flagged Events", f"{len(anom_df):,}", "Across 10 Business & Statistical Types")
    with col2:
        so_count = (anom_df["Anomaly_Type"] == "Stockout Constrained").sum()
        render_kpi_card("Stockout Constraints", f"{so_count:,}", "Unconstrained Demand Reconstructed")
    with col3:
        sb_count = anom_df["Anomaly_Type"].str.contains("Structural Break|Disruption|Market Entry|ERP").sum()
        render_kpi_card("Breaks & Disruptions", f"{sb_count:,}", "COVID, Chips, 2022 Inflation, Red Sea")
    with col4:
        eb_count = (anom_df["Anomaly_Type"] == "Early Buying (Pre-Price Hike)").sum()
        render_kpi_card("Early Buying Surges", f"{eb_count:,}", "Nov/Dec Pre-Hike Pull-Forwards")

    st.markdown("---")

    # 2. Anomaly Timeline Scatter Chart
    st.markdown("#### 📈 Historical Demand Timeline with Flagged Business Anomalies")
    monthly_units = train_df.groupby("Date")["Units_Sold_Unconstrained"].sum().reset_index()

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=monthly_units["Date"], y=monthly_units["Units_Sold_Unconstrained"],
        mode="lines", name="Portfolio Unconstrained Demand", line=dict(color="#64748B", width=2)
    ))

    color_map = {
        "Stockout Constrained": "#EF4444",
        "Structural Break (COVID-19)": "#8B5CF6",
        "Structural Break (Chip Shortage)": "#EC4899",
        "Structural Break (2022 Inflation & Surcharge)": "#F97316",
        "Early Buying (Pre-Price Hike)": "#F59E0B",
        "Holiday Dip (European Summer)": "#06B6D4",
        "Market Entry (MEA Opening)": "#10B981",
        "System Event (ERP Migration)": "#EAB308",
        "Customer Disruption (Nexon Strike)": "#F43F5E",
        "Logistics Disruption (Red Sea)": "#3B82F6",
        "Customer Capacity Expansion (Helios Line 2)": "#22C55E",
        "Statistical Outlier (One-Off Spike)": "#A855F7",
    }

    anom_agg = anom_df.groupby(["Date", "Anomaly_Type"]).agg(Count=("SKU_ID", "count")).reset_index()
    anom_agg = anom_agg.merge(monthly_units, on="Date", how="inner")

    for atype, grp in anom_agg.groupby("Anomaly_Type"):
        fig_anom.add_trace(go.Scatter(
            x=grp["Date"], y=grp["Units_Sold_Unconstrained"],
            mode="markers", name=atype,
            marker=dict(size=9, color=color_map.get(atype, "#00E5FF"), symbol="diamond"),
            hovertemplate=f"<b>%{{x}}</b><br>Type: {atype}<br>Flagged Records: %{{text}}<extra></extra>",
            text=grp["Count"],
        ))

    fig_anom.update_layout(
        title="Portfolio Demand with Highlighted Anomalies (Click Legend to Filter Categories)",
        template="plotly_dark", paper_bgcolor="#1E293B", plot_bgcolor="#0F172A",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    render_chart(fig_anom)

    # 3. Tabs for Anomaly Register, Switches, Lost Accounts
    tab_reg, tab_sw, tab_lost = st.tabs(["📋 Anomaly Register", "🔁 Customer-SKU Switches", "📉 Lost / Dormant Accounts"])
    with tab_reg:
        type_filter = st.selectbox("Filter Anomaly Category:", ["All"] + sorted(anom_df["Anomaly_Type"].unique().tolist()))
        display_df = anom_df if type_filter == "All" else anom_df[anom_df["Anomaly_Type"] == type_filter]
        avail_cols = [c for c in ["Date", "SKU_ID", "Region", "Customer_Name", "Anomaly_Type", "Units_Sold", "Units_Sold_Unconstrained", "Lost_Units", "Reason", "Action_Taken"] if c in display_df.columns]
        render_table(display_df[avail_cols].head(250))
        st.caption(f"Displaying {min(250, len(display_df))} of {len(display_df):,} flagged records.")

    with tab_sw:
        sw_path = REPORTS_DIR / "customer_switches.csv"
        if sw_path.exists():
            render_table(pd.read_csv(sw_path).head(150))

    with tab_lost:
        lost_path = REPORTS_DIR / "lost_accounts.csv"
        if lost_path.exists():
            render_table(pd.read_csv(lost_path).head(150))
