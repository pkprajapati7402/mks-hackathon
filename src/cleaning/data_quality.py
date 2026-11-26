"""
Data Quality Report Generator.
Generates machine-readable data_quality_report.json and human-readable data_quality_report.md.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.config import REPORTS_DIR

def generate_data_quality_report(
    train_dq: Dict[str, Any],
    sku_dq: Dict[str, Any],
    macro_dq: Dict[str, Any],
    events_dq: Dict[str, Any],
    stockouts_dq: Dict[str, Any],
    audit_summary: pd.DataFrame,
    output_dir: Path = REPORTS_DIR
) -> Dict[str, Any]:
    """Compile and export comprehensive data quality audit report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "metadata": {
            "title": "Celeris Automobili Demand Forecasting Data Quality & Audit Report",
            "version": "1.0.0",
            "executive_summary": "Complete ERP data cleaning and audit trail covering 2019-01 to 2026-10 history."
        },
        "file_summaries": {
            "train.csv": {
                "raw_rows": train_dq.get("raw_rows", 0),
                "clean_rows": train_dq.get("clean_rows", 0),
                "duplicates_removed": train_dq.get("T4_duplicate_record_ids_removed", 0) + train_dq.get("T4b_duplicate_keys_removed", 0),
                "dates_parsed": train_dq.get("T1_dates_parsed", 0),
                "year_2062_typos_fixed": train_dq.get("T3_2062_typos_fixed", 0),
                "skus_normalized": train_dq.get("T8_skus_normalized", 0),
                "customers_normalized": train_dq.get("T10_customers_normalized", 0),
                "channels_normalized": train_dq.get("T6_channels_normalized", 0),
                "blank_regions_recovered": train_dq.get("T5_blank_regions_recovered", 0),
                "negative_units_reversals": train_dq.get("T11_negative_units_reversals", 0),
                "blank_units_filled": train_dq.get("T11_blank_units_filled_stockouts", 0),
                "entry_errors_corrected": train_dq.get("A3_entry_errors_fixed", 0),
                "unit_10x_recovered": train_dq.get("A3_unit_10x_recovered", 0),
                "unit_100x_recovered": train_dq.get("A3_unit_100x_recovered", 0),
                "prices_repaired": train_dq.get("T12_prices_repaired", 0),
                "cents_100x_fixed": train_dq.get("T12_cents_100x_fixed", 0),
                "decimal_10x_fixed": train_dq.get("T12_decimal_10x_fixed", 0),
                "currency_eur_fixed": train_dq.get("T12_currency_eur_fixed", 0),
                "revenue_reconciled": train_dq.get("T13_revenue_reconciled", 0),
                "family_mismatches_fixed": train_dq.get("T7_family_mismatches_fixed", 0),
                "bu_mismatches_fixed": train_dq.get("T7_bu_mismatches_fixed", 0),
                "lifecycle_violations_zeroed": train_dq.get("R5_lifecycle_violations_zeroed", 0),
                "clean_total_units": train_dq.get("clean_total_units", 0),
                "clean_total_revenue_usd": train_dq.get("clean_total_revenue_usd", 0.0)
            },
            "sku_master.csv": {
                "raw_rows": sku_dq.get("raw_rows", 0),
                "clean_rows": sku_dq.get("clean_rows", 0),
                "bu_conflicts_repaired": sku_dq.get("S1_conflict_count", 0),
                "missing_replaces_sku_references": sku_dq.get("S2_missing_replaces_sku", 0)
            },
            "macro_indicators.csv": {
                "raw_rows": macro_dq.get("raw_rows", 0),
                "clean_rows": macro_dq.get("clean_rows", 0),
                "duplicate_months_resolved": macro_dq.get("M1_duplicate_months", 0),
                "fraction_pct_fixed": macro_dq.get("M2_fraction_pct_fixed", 0),
                "pmi_decimal_shifts_fixed": macro_dq.get("M3_pmi_decimal_shifted", 0)
            },
            "events_calendar.csv": {
                "raw_rows": events_dq.get("raw_rows", 0),
                "clean_rows": events_dq.get("clean_rows", 0)
            },
            "inventory_stockouts.csv": {
                "raw_rows": stockouts_dq.get("raw_rows", 0),
                "clean_rows": stockouts_dq.get("clean_rows", 0),
                "lost_units_reconciled": stockouts_dq.get("lost_units_reconciled", 0)
            }
        },
        "audit_rule_summary": audit_summary.to_dict(orient="records") if not audit_summary.empty else []
    }

    # Save JSON report
    json_path = output_dir / "data_quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown report
    md_path = output_dir / "data_quality_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 📋 Celeris Automobili Industrial — Data Quality & Integrity Audit Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write("This report documents the automated data cleaning and quality audit pipeline executed for the **MKS AIML Hackathon Phase 2 Demand Forecasting Control Tower**. ")
        f.write("All raw ERP extracts spanning **2019-01 to 2026-10** have been systematically reconciled, deduplicated, and normalized against the reference master data.\n\n")
        
        f.write("## 2. File Processing Summaries\n\n")
        f.write("| File | Raw Rows | Clean Rows | Key Anomalies Addressed |\n")
        f.write("|---|---|---|---|\n")
        dup_total = train_dq.get("T4_duplicate_record_ids_removed", 0) + train_dq.get("T4b_duplicate_keys_removed", 0)
        f.write(f"| `train.csv` | {train_dq.get('raw_rows', 0):,} | {train_dq.get('clean_rows', 0):,} | Deduplicated {dup_total:,} records; fixed 2062 typos ({train_dq.get('T3_2062_typos_fixed', 0)}); repaired {train_dq.get('T12_prices_repaired', 0):,} prices; filled {train_dq.get('T11_blank_units_filled_stockouts', 0):,} stockouts |\n")
        f.write(f"| `sku_master.csv` | {sku_dq.get('raw_rows', 0):,} | {sku_dq.get('clean_rows', 0):,} | Repaired {sku_dq.get('S1_conflict_count', 0)} master BU conflict(s) |\n")
        f.write(f"| `macro_indicators.csv` | {macro_dq.get('raw_rows', 0):,} | {macro_dq.get('clean_rows', 0):,} | Resolved duplicate 2022-06; repaired {macro_dq.get('M2_fraction_pct_fixed', 0)} fraction percentages and {macro_dq.get('M3_pmi_decimal_shifted', 0)} PMI shifts |\n")
        f.write(f"| `events_calendar.csv` | {events_dq.get('raw_rows', 0):,} | {events_dq.get('clean_rows', 0):,} | Cleaned mixed date formats and parsed Customer/Region composite scopes |\n")
        f.write(f"| `inventory_stockouts.csv` | {stockouts_dq.get('raw_rows', 0):,} | {stockouts_dq.get('clean_rows', 0):,} | Standardized keys and verified balance equation across {stockouts_dq.get('clean_rows', 0):,} stockout records |\n\n")

        f.write("## 3. Data Cleaning Rule Catalog\n\n")
        f.write("- **T1 (Date Normalization)**: Standardized heterogeneous date formats (`YYYY-MM`, `YYYY-MM-DD`, `MM/YYYY`, `Mon-YYYY`) into ISO standard `YYYY-MM`.\n")
        f.write("- **T2 (Year/Month Reconcile)**: Enforced authoritative `Date` precedence per Rule 1.\n")
        f.write("- **T3 (2062 Digit Transposition)**: Corrected OCR/data-entry digit swap of `2062` to `2026`.\n")
        f.write("- **T4 (Record_ID Deduplication)**: Kept latest `Load_Timestamp` row per duplicate ERP transaction (Rule 2).\n")
        f.write("- **T5 (Region Recovery)**: Mapped region aliases to 6 canonical regions and recovered blank regions via customer historical affinity.\n")
        f.write("- **T6 (Sales Channel Mapping)**: Standardized channel labels to 4 canonical categories.\n")
        f.write("- **T7 & T8 (SKU Normalization)**: Normalized irregular SKU identifiers to `[PREFIX]-[NNN]` and aligned Product Family / BU strictly with master data.\n")
        f.write(f"- **T10 (Customer Normalization)**: Reconciled {train_dq.get('T10_customers_normalized', 0):,} noisy ERP customer entries to 12 canonical tier-1 accounts.\n")
        f.write("- **T11 & A2 (Units Sold & Stockout Fill)**: Handled returns/reversals, removed thousand-separators, and filled unfulfilled orders from stockout records.\n")
        f.write("- **T12 & T13 (Price & Revenue Integrity)**: Repaired currency glitches, zero prices, and 100x pence errors using annual median USD prices; strictly reconciled `Revenue_USD = Units_Sold * Unit_Price_USD`.\n")
        f.write("- **R5 (Lifecycle Boundaries)**: Enforced zero sales prior to `Launch_Date` and after `EOL_Date` (Rule 5).\n\n")

        f.write("## 4. Audit Modifications Breakdown\n\n")
        if not audit_summary.empty:
            f.write(audit_summary.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No audit modifications recorded.\n\n")

        f.write("> [!NOTE]\n")
        f.write("> Machine-readable quality metrics are saved in `reports/data_quality_report.json` and row-level edits in `reports/audit_trail.parquet`.\n")

    return report_data
