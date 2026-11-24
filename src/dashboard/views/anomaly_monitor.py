"""
Anomaly Monitor View (Required View 4).
Displays interactive timeline scatter plot of flagged anomalies,
anomaly categorization breakdown, and searchable Anomaly Register.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import REPORTS_DIR
from src.dashboard.components import render_kpi_card

def render_view(train_df: pd.DataFrame):
    st.markdown("### 🚨 Anomaly Monitor & Anomaly Register")
    st.markdown("Automated detection, categorization, and tracking of outliers, structural breaks, stockouts, early buying, and seasonal dips.")

    reg_path = REPORTS_DIR / "anomaly_register.csv"
    if not reg_path.exists():
        st.warning("Anomaly register not found. Run pipeline first.")
        return

    anom_df = pd.read_csv(reg_path)

    # 1. Headline KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Flagged Events", f"{len(anom_df):,}", "In Cleaned Demand History")
    with col2:
        so_count = (anom_df["Anomaly_Type"] == "Stockout Constrained").sum()
        render_kpi_card("Stockout Months", f"{so_count:,}", "Corrected via Inventory Records")
    with col3:
        sb_count = anom_df["Anomaly_Type"].str.contains("Structural Break").sum()
        render_kpi_card("Structural Breaks", f"{sb_count:,}", "COVID-19 & Chip Shortage")
    with col4:
        eb_count = (anom_df["Anomaly_Type"] == "Early Buying (Pre-Price Hike)").sum()
        render_kpi_card("Early Buying Surges", f"{eb_count:,}", "Pre-January Price Pull-Forward")

    st.markdown("---")

    # 2. Anomaly Timeline Scatter Chart
    st.markdown("#### 📈 Historical Timeline with Flagged Anomaly Points")
    monthly_units = train_df.groupby("Date")["Units_Sold_Unconstrained"].sum().reset_index()

    fig_anom = go.Figure()
    # Baseline line
    fig_anom.add_trace(go.Scatter(
        x=monthly_units["Date"],
        y=monthly_units["Units_Sold_Unconstrained"],
        mode="lines",
        name="Portfolio Demand",
        line=dict(color="#475569", width=2)
    ))

    # Overlay anomaly points
    color_map = {
        "Stockout Constrained": "#EF4444",
        "Structural Break (COVID-19)": "#8B5CF6",
        "Structural Break (Chip Shortage)": "#EC4899",
        "Early Buying (Pre-Price Hike)": "#F59E0B",
        "Holiday Dip (European Summer)": "#06B6D4"
    }

    # Aggregate anomalies by Date and Type
    anom_agg = anom_df.groupby(["Date", "Anomaly_Type"]).agg(
        Count=("SKU_ID", "count"),
        Total_Units=("Units_Sold", "sum")
    ).reset_index()

    # Merge with monthly total for positioning
    anom_agg = anom_agg.merge(monthly_units, on="Date")

    for atype, grp in anom_agg.groupby("Anomaly_Type"):
        fig_anom.add_trace(go.Scatter(
            x=grp["Date"],
            y=grp["Units_Sold_Unconstrained"],
            mode="markers",
            name=atype,
            marker=dict(
                size=10,
                color=color_map.get(atype, "#00E5FF"),
                symbol="diamond"
            ),
            hovertemplate="<b>%{x}</b><br>Type: " + atype + "<br>Affected SKUs: %{text}<extra></extra>",
            text=grp["Count"]
        ))

    fig_anom.update_layout(
        title="Portfolio Unconstrained Demand with Highlighted Macro/Supply Anomalies",
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    # 3. Categorization Bar Chart & Searchable Register Table
    st.markdown("#### 📋 Searchable Anomaly Register")
    type_filter = st.selectbox("Filter Anomaly Category:", ["All"] + sorted(anom_df["Anomaly_Type"].unique().tolist()))
    display_df = anom_df if type_filter == "All" else anom_df[anom_df["Anomaly_Type"] == type_filter]

    avail_cols = [c for c in ["Date", "SKU_ID", "Region", "Customer_Name", "Anomaly_Type", "Units_Sold", "Units_Sold_Unconstrained", "Lost_Units", "Reason", "Action_Taken"] if c in display_df.columns]
    st.dataframe(
        display_df[avail_cols].head(250),
        use_container_width=True
    )
    st.caption(f"Displaying {min(250, len(display_df))} of {len(display_df):,} flagged records.")
