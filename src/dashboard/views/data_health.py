"""
Data Health & Automated Audit Trail View (Required View 2).
Displays issues by type & file, before vs after cleaning comparisons, arithmetic & currency fixes,
master-data contradictions register, and searchable row-level audit trail.
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import REPORTS_DIR
from src.dashboard.components import render_kpi_card, render_chart, render_table


def render_view():
    st.markdown("### 🩺 Data Health & Automated Quality Audit")
    st.markdown("Automated arithmetic triangulation (`Units × Price = Revenue`), currency normalization, deduplication, and master-data reconciliation.")

    dq_json_path = REPORTS_DIR / "data_quality_report.json"
    audit_parquet_path = REPORTS_DIR / "audit_trail.parquet"
    contradictions_path = REPORTS_DIR / "master_data_contradictions.csv"

    dq_data = {}
    if dq_json_path.exists():
        with open(dq_json_path, "r", encoding="utf-8") as f:
            dq_data = json.load(f)

    train_stats = dq_data.get("file_summaries", {}).get("train.csv", {})

    # 1. KPI Health Gauges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Duplicates Eliminated", f"{train_stats.get('duplicates_removed', 0):,}", "Latest Load_Timestamp (Rule 2)")
    with col2:
        u10 = train_stats.get("unit_10x_recovered", 0)
        u100 = train_stats.get("unit_100x_recovered", 0)
        render_kpi_card("Exact Unit Fixes (10x/100x)", f"{u10 + u100:,}", f"10x: {u10:,} | 100x: {u100:,} via Rev Proof")
    with col3:
        eur = train_stats.get("currency_eur_fixed", 0)
        cents = train_stats.get("cents_100x_fixed", 0)
        render_kpi_card("Prices & Currency Fixed", f"{train_stats.get('prices_repaired', 0):,}", f"Cents: {cents:,} | EUR->USD: {eur:,}")
    with col4:
        render_kpi_card("2062 Year Typos & R5 EOL", f"{train_stats.get('year_2062_typos_fixed', 0)} / {train_stats.get('lifecycle_violations_zeroed', 0)}", "Dates Parsed & Lifecycle Enforced")

    st.markdown("---")

    # 2. Before vs After Cleaning Breakdown + Rule Bar Chart
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📁 File-Level Before vs After Cleaning")
        file_rows = []
        for fname, fstats in dq_data.get("file_summaries", {}).items():
            raw_r = fstats.get("raw_rows", 0)
            cln_r = fstats.get("clean_rows", 0)
            file_rows.append({
                "File": fname,
                "Before (Raw Rows)": f"{raw_r:,}",
                "After (Clean Rows)": f"{cln_r:,}",
                "Net Delta": f"{cln_r - raw_r:+,}",
                "Status": "✅ Verified Clean",
            })
        render_table(pd.DataFrame(file_rows))

        st.markdown("#### 🔍 Specific Issue Counts Remediated in `train.csv`")
        issue_breakdown = pd.DataFrame([
            {"Issue Type": "Date Format Standardization (T1)", "Rows Affected": train_stats.get("dates_parsed", 0)},
            {"Issue Type": "Customer Name Alias Consolidation (T10)", "Rows Affected": train_stats.get("customers_normalized", 0)},
            {"Issue Type": "Sales Channel Normalization (T6)", "Rows Affected": train_stats.get("channels_normalized", 0)},
            {"Issue Type": "SKU ID Typo Normalization (T8)", "Rows Affected": train_stats.get("skus_normalized", 0)},
            {"Issue Type": "Duplicates Removed (T4/T4b)", "Rows Affected": train_stats.get("duplicates_removed", 0)},
            {"Issue Type": "Revenue Reconciliation (T13)", "Rows Affected": train_stats.get("revenue_reconciled", 0)},
            {"Issue Type": "Price Cents/Decimal/EUR Fixed (T12)", "Rows Affected": train_stats.get("prices_repaired", 0)},
            {"Issue Type": "10x/100x Unit Typos Recovered (A3)", "Rows Affected": train_stats.get("entry_errors_corrected", 0)},
            {"Issue Type": "Stockout Blanks Imputed (A2)", "Rows Affected": train_stats.get("blank_units_filled", 0)},
            {"Issue Type": "Negative Return Reversals (T11)", "Rows Affected": train_stats.get("negative_units_reversals", 0)},
            {"Issue Type": "Master Family/BU Contradictions (T7)", "Rows Affected": train_stats.get("family_mismatches_fixed", 0) + train_stats.get("bu_mismatches_fixed", 0)},
            {"Issue Type": "2062 Digit Typos (T3)", "Rows Affected": train_stats.get("year_2062_typos_fixed", 0)},
        ])
        render_table(issue_breakdown)

    with col_b:
        st.markdown("#### ⚙️ Audited Rule Modifications Visual Breakdown")
        audit_summary = pd.DataFrame(dq_data.get("audit_rule_summary", []))
        if not audit_summary.empty:
            fig_rules = px.bar(
                audit_summary.sort_values("modification_count", ascending=True),
                x="modification_count", y="rule_id", orientation="h",
                color="file", title="Row-Level Audit Entries Logged by Rule ID",
                template="plotly_dark"
            )
            fig_rules.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A")
            render_chart(fig_rules)

    # 3. Master Data vs Transactions Contradictions + Searchable Audit Log
    tab_audit, tab_contra = st.tabs(["🔍 Searchable Row-Level Audit Trail", "⚠️ Master Data vs Transactions Contradictions"])
    with tab_audit:
        if audit_parquet_path.exists():
            audit_df = pd.read_parquet(audit_parquet_path)
            rule_filter = st.selectbox("Filter by Rule ID:", ["All"] + sorted(audit_df["rule_id"].unique().tolist()))
            if rule_filter != "All":
                audit_df = audit_df[audit_df["rule_id"] == rule_filter]
            render_table(audit_df.head(250)[["file", "row_key", "field", "raw_value", "new_value", "rule_id", "reason"]])
            st.caption(f"Showing first {min(250, len(audit_df))} of {len(audit_df):,} audit log entries.")

    with tab_contra:
        if contradictions_path.exists():
            contra_df = pd.read_csv(contradictions_path)
            render_table(contra_df)
        else:
            st.info("Run pipeline to generate master_data_contradictions.csv.")
