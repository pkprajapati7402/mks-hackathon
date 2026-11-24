"""
Data Export & Report Center View (Optional View 7).
Enables instant download of competition submission files, cleaned datasets,
anomaly registers, prediction intervals, and C-suite reports.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from src.config import OUTPUTS_DIR, REPORTS_DIR, CLEANED_DATA_DIR, PROJECT_ROOT
from src.dashboard.components import render_kpi_card

def render_view():
    st.markdown("### 💾 Data Export & Governance Report Center")
    st.markdown("One-click download of competition deliverables, reconciled forecasts, machine-readable audits, and executive summaries.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Competition Submission", "58,219 Rows", "Date, SKU_ID, Region, Units")
    with col2:
        render_kpi_card("Cleaned History", "244,490 Rows", "Parquet Compressed")
    with col3:
        render_kpi_card("Governance Reports", "JSON + Markdown", "Full Audit Integrity")
    with col4:
        render_kpi_card("Forecast Intervals", "P10, P50, P90", "Full Distribution Data")

    st.markdown("---")

    st.markdown("#### 📥 Instant File Downloads")

    # 1. Submission File
    sub_path = OUTPUTS_DIR / "submission.csv"
    if sub_path.exists():
        with open(sub_path, "rb") as f:
            st.download_button(
                label="📄 Download Official submission.csv (58,219 rows)",
                data=f.read(),
                file_name="submission.csv",
                mime="text/csv",
                type="primary"
            )

    col_a, col_b = st.columns(2)

    with col_a:
        # Anomaly Register
        anom_path = REPORTS_DIR / "anomaly_register.csv"
        if anom_path.exists():
            with open(anom_path, "rb") as f:
                st.download_button(
                    label="🚨 Download anomaly_register.csv",
                    data=f.read(),
                    file_name="anomaly_register.csv",
                    mime="text/csv"
                )

        # Data Quality Report JSON
        dq_path = REPORTS_DIR / "data_quality_report.json"
        if dq_path.exists():
            with open(dq_path, "rb") as f:
                st.download_button(
                    label="📋 Download data_quality_report.json (Machine-Readable)",
                    data=f.read(),
                    file_name="data_quality_report.json",
                    mime="application/json"
                )

        # Model Card
        card_path = REPORTS_DIR / "model_card.md"
        if card_path.exists():
            with open(card_path, "rb") as f:
                st.download_button(
                    label="📑 Download model_card.md (ML Model Governance)",
                    data=f.read(),
                    file_name="model_card.md",
                    mime="text/markdown"
                )

    with col_b:
        # Backtest Metrics
        bt_path = OUTPUTS_DIR / "backtest_metrics.csv"
        if bt_path.exists():
            with open(bt_path, "rb") as f:
                st.download_button(
                    label="🎯 Download backtest_metrics.csv",
                    data=f.read(),
                    file_name="backtest_metrics.csv",
                    mime="text/csv"
                )

        # Executive Summary
        exec_path = REPORTS_DIR / "executive_summary.md"
        if exec_path.exists():
            with open(exec_path, "rb") as f:
                st.download_button(
                    label="🏛️ Download executive_summary.md (Strategic Outlook)",
                    data=f.read(),
                    file_name="executive_summary.md",
                    mime="text/markdown"
                )

        # Customer Ordering Cycles
        cycle_path = REPORTS_DIR / "customer_ordering_cycles.csv"
        if cycle_path.exists():
            with open(cycle_path, "rb") as f:
                st.download_button(
                    label="🔄 Download customer_ordering_cycles.csv",
                    data=f.read(),
                    file_name="customer_ordering_cycles.csv",
                    mime="text/csv"
                )
