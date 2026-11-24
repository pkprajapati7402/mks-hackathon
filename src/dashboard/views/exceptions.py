"""
Exception List for Planners View (Optional View 6).
Generates automated actionable alerts for demand planners:
imminent EOL cutoffs, extreme forecast uncertainty, dormant accounts, and supply risks.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from src.config import REPORTS_DIR
from src.dashboard.components import render_kpi_card, render_table

def render_view(forecast_df: pd.DataFrame, sku_master: pd.DataFrame):
    st.markdown("### 📋 Planner Exception Center & Operational Alerts")
    st.markdown("Automated algorithmic alerts highlighting high-risk SKUs, imminent EOL transitions, volatile demand intervals, and customer churn.")

    # 1. Generate Exceptions
    exceptions = []

    # A. Imminent EOL Cutoffs (< 6 months to EOL)
    eol_active = sku_master[sku_master["EOL_Date"].notna() & (sku_master["EOL_Date"] >= "2026-11") & (sku_master["EOL_Date"] <= "2027-05")]
    for _, row in eol_active.iterrows():
        exceptions.append({
            "Severity": "🔴 High",
            "Category": "Catalog Transition",
            "Entity": row["SKU_ID"],
            "Details": f"Scheduled EOL in {row['EOL_Date']}. Predecessor mapping: {row.get('Replaces_SKU', 'None')}",
            "Recommended_Action": "Initiate inventory ramp-down and shift customer allocations to successor SKU"
        })

    # B. High Uncertainty SKUs (P90 / P10 > 3.0 with meaningful volume)
    sku_agg = forecast_df.groupby(["SKU_ID", "Product_Family", "Region"]).agg({
        "P10": "sum",
        "P50": "sum",
        "P90": "sum"
    }).reset_index()

    high_uncertainty = sku_agg[(sku_agg["P50"] > 300) & (sku_agg["P90"] > 3.0 * np.maximum(sku_agg["P10"], 1.0))]
    for _, row in high_uncertainty.head(25).iterrows():
        exceptions.append({
            "Severity": "🟡 Medium",
            "Category": "High Forecast Volatility",
            "Entity": f"{row['SKU_ID']} ({row['Region']})",
            "Details": f"Wide prediction interval: P10={row['P10']:,.0f}, P50={row['P50']:,.0f}, P90={row['P90']:,.0f} units",
            "Recommended_Action": "Review sales order pipeline with regional commercial team; buffer safety stock to P75"
        })

    # C. Dormant Customer Accounts
    lost_path = REPORTS_DIR / "lost_accounts.csv"
    if lost_path.exists():
        lost_df = pd.read_csv(lost_path)
        for _, row in lost_df.head(25).iterrows():
            exceptions.append({
                "Severity": "🟡 Medium",
                "Category": "Customer Churn Risk",
                "Entity": f"{row['Customer_Name']} - {row['SKU_ID']}",
                "Details": f"Zero orders for >6 consecutive months (Prior volume: {row['Historical_Units']:,} units)",
                "Recommended_Action": "Account executive outreach to verify contract renewal status"
            })

    # D. Zero Forecast Warning on Active SKUs
    active_zero = forecast_df.groupby("SKU_ID")["Units_Sold_Forecast"].sum().reset_index()
    active_zero = active_zero[active_zero["Units_Sold_Forecast"] == 0]
    active_zero = active_zero.merge(sku_master[["SKU_ID", "Lifecycle_Status"]], on="SKU_ID")
    active_zero = active_zero[active_zero["Lifecycle_Status"] == "Active"]
    for _, row in active_zero.head(15).iterrows():
        exceptions.append({
            "Severity": "🔴 High",
            "Category": "Zero Demand on Active SKU",
            "Entity": row["SKU_ID"],
            "Details": "SKU is marked Active in master catalog but has 0 units projected over entire 19-month horizon",
            "Recommended_Action": "Verify commercial contract status or reclassify SKU to Discontinued"
        })

    ex_df = pd.DataFrame(exceptions, columns=["Severity", "Category", "Entity", "Details", "Recommended_Action"])

    # 2. KPI Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Active Exceptions", f"{len(ex_df):,}", "Pending Planner Action")
    with col2:
        high_sev = (ex_df["Severity"] == "🔴 High").sum() if not ex_df.empty else 0
        render_kpi_card("High Severity Items", f"{high_sev}", "Require Immediate Resolution")
    with col3:
        med_sev = (ex_df["Severity"] == "🟡 Medium").sum() if not ex_df.empty else 0
        render_kpi_card("Medium Severity Items", f"{med_sev}", "Monitoring / Weekly Review")
    with col4:
        render_kpi_card("Resolution Protocol", "Automated Playbooks", "Aligned with S&OP Guidelines")

    st.markdown("---")

    if ex_df.empty:
        st.success("No exceptions flagged for the currently filtered selection.")
        return

    # 3. Searchable Exceptions Table
    st.markdown("#### 🚨 Prioritized Planner Action Items")
    sev_filter = st.selectbox("Filter by Severity:", ["All", "🔴 High", "🟡 Medium"])
    cat_filter = st.selectbox("Filter by Category:", ["All"] + sorted(ex_df["Category"].unique().tolist()))

    filtered_ex = ex_df
    if sev_filter != "All":
        filtered_ex = filtered_ex[filtered_ex["Severity"] == sev_filter]
    if cat_filter != "All":
        filtered_ex = filtered_ex[filtered_ex["Category"] == cat_filter]

    render_table(filtered_ex)
