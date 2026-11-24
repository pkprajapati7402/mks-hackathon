# 📋 Celeris Automobili Industrial — Data Quality & Integrity Audit Report

## 1. Executive Summary
This report documents the automated data cleaning and quality audit pipeline executed for the **MKS AIML Hackathon Phase 2 Demand Forecasting Control Tower**. All raw ERP extracts spanning **2019-01 to 2026-10** have been systematically reconciled, deduplicated, and normalized against the reference master data.

## 2. File Processing Summaries

| File | Raw Rows | Clean Rows | Key Anomalies Addressed |
|---|---|---|---|
| `train.csv` | 248,248 | 244,490 | Deduplicated 3,682 records; fixed 2062 typos (30); repaired 37,351 prices; filled 90 stockouts |
| `sku_master.csv` | 601 | 601 | Repaired 1 Precision Machined Parts BU conflicts |
| `macro_indicators.csv` | 114 | 113 | Resolved duplicate 2022-06; repaired 12 fraction percentages and 1 PMI shifts |
| `events_calendar.csv` | 68 | 68 | Cleaned mixed date formats and parsed Customer/Region composite scopes |
| `inventory_stockouts.csv` | 5,830 | 5,830 | Standardized keys and verified balance equation across 5,830 stockout records |

## 3. Data Cleaning Rule Catalog

- **T1 (Date Normalization)**: Standardized heterogeneous date formats (`YYYY-MM`, `YYYY-MM-DD`, `MM/YYYY`, `Mon-YYYY`) into ISO standard `YYYY-MM`.
- **T2 (Year/Month Reconcile)**: Enforced authoritative `Date` precedence per Rule 1.
- **T3 (2062 Digit Transposition)**: Corrected OCR/data-entry digit swap of `2062` to `2026`.
- **T4 (Record_ID Deduplication)**: Kept latest `Load_Timestamp` row per duplicate ERP transaction (Rule 2).
- **T5 (Region Recovery)**: Mapped region aliases to 6 canonical regions and recovered blank regions via customer historical affinity.
- **T6 (Sales Channel Mapping)**: Standardized channel labels to 4 canonical categories.
- **T7 & T8 (SKU Normalization)**: Normalized irregular SKU identifiers to `[PREFIX]-[NNN]` and aligned Product Family / BU strictly with master data.
- **T10 (Customer Normalization)**: Reconciled 220 noisy ERP customer variants to 12 canonical tier-1 accounts.
- **T11 & A2 (Units Sold & Stockout Fill)**: Handled returns/reversals, removed thousand-separators, and filled unfulfilled orders from stockout records.
- **T12 & T13 (Price & Revenue Integrity)**: Repaired currency glitches, zero prices, and 100x pence errors using annual median USD prices; strictly reconciled `Revenue_USD = Units_Sold * Unit_Price_USD`.
- **R5 (Lifecycle Boundaries)**: Enforced zero sales prior to `Launch_Date` and after `EOL_Date` (Rule 5).

## 4. Audit Modifications Breakdown

| file                 | rule_id   | field                   |   modification_count |
|:---------------------|:----------|:------------------------|---------------------:|
| macro_indicators.csv | M1        | Row_Deduplication       |                    1 |
| macro_indicators.csv | M2        | Auto_Production_YoY_Pct |                   12 |
| macro_indicators.csv | M3        | ISM_Manufacturing_PMI   |                    1 |
| macro_indicators.csv | M4        | Missing_Values          |                    5 |
| sku_master.csv       | S1        | Business_Unit           |                    1 |
| train.csv            | A2        | Units_Sold              |                   90 |
| train.csv            | A3        | Units_Sold              |                  107 |
| train.csv            | R5        | Units_Sold              |                   15 |
| train.csv            | T12       | Unit_Price_USD          |                  100 |
| train.csv            | T3        | Date                    |                   30 |
| train.csv            | T5        | Region                  |                  942 |

> [!NOTE]
> Machine-readable quality metrics are saved in `reports/data_quality_report.json` and row-level edits in `reports/audit_trail.parquet`.
