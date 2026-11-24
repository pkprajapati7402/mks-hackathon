#!/usr/bin/env python
"""
Celeris Automobili Demand Forecasting & Planning Control Tower
One-Command Master Pipeline Runner (Phase 2 AIML Hackathon).
Executes end-to-end cleaning, anomaly detection, feature engineering,
rolling-origin backtesting, hierarchical reconciliation, and reporting.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_TRAIN, RAW_TEST, RAW_MACRO, RAW_SKU_MASTER, RAW_EVENTS, RAW_STOCKOUTS,
    CLEANED_DATA_DIR, OUTPUTS_DIR, REPORTS_DIR
)
from src.audit import audit
from src.cleaning.clean_auxiliary import (
    clean_sku_master, clean_macro_indicators, clean_events_calendar, clean_inventory_stockouts
)
from src.cleaning.clean_train import clean_train_data
from src.cleaning.data_quality import generate_data_quality_report
from src.anomalies.unconstrained import reconstruct_unconstrained_demand
from src.anomalies.detector import detect_anomalies
from src.anomalies.customer_cycles import analyze_customer_cycles
from src.features.feature_pipeline import build_feature_store
from src.models.backtester import run_rolling_origin_backtest
from src.models.family_selector import select_best_model_per_family
from src.models.lightgbm_engine import train_and_forecast_lgb
from src.models.cold_start import apply_lifecycle_and_cold_start
from src.models.reconciliation import reconcile_and_generate_submission
from src.reports_generator import generate_model_card, generate_executive_summary

def run_pipeline(stage: str = "all"):
    print("=" * 80)
    print("  CELERIS AUTOMOBILI INDUSTRIAL: DEMAND FORECASTING CONTROL TOWER")
    print("  MKS AIML Hackathon Phase 2 Production Pipeline")
    print("=" * 80)

    t0_master = time.time()

    # ---------------------------------------------------------
    # STAGE 1: LOAD & CLEAN DATASETS
    # ---------------------------------------------------------
    print("\n>>> [STAGE 1/6] Loading & Cleaning Raw ERP Data and Auxiliary Tables...")
    t0 = time.time()

    raw_train = pd.read_csv(RAW_TRAIN, low_memory=False)
    raw_test = pd.read_csv(RAW_TEST, low_memory=False)
    raw_sku = pd.read_csv(RAW_SKU_MASTER)
    raw_macro = pd.read_csv(RAW_MACRO)
    raw_events = pd.read_csv(RAW_EVENTS)
    raw_stockouts = pd.read_csv(RAW_STOCKOUTS)

    # Clean auxiliary tables
    sku_master_clean, sku_dq = clean_sku_master(raw_sku)
    macro_clean, macro_dq = clean_macro_indicators(raw_macro)
    events_clean, events_dq = clean_events_calendar(raw_events)
    stockouts_clean, stockouts_dq = clean_inventory_stockouts(raw_stockouts)

    # Clean train dataset
    train_clean, train_dq = clean_train_data(raw_train, sku_master_clean, stockouts_clean)

    # Save cleaned tables
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sku_master_clean.to_csv(CLEANED_DATA_DIR / "sku_master.csv", index=False)
    macro_clean.to_csv(CLEANED_DATA_DIR / "macro_indicators.csv", index=False)
    events_clean.to_csv(CLEANED_DATA_DIR / "events_calendar.csv", index=False)
    stockouts_clean.to_csv(CLEANED_DATA_DIR / "inventory_stockouts.csv", index=False)
    train_clean.to_parquet(CLEANED_DATA_DIR / "clean_train.parquet", index=False)

    # Generate Data Quality Report
    audit_summary = audit.summary()
    audit.save()
    generate_data_quality_report(
        train_dq, sku_dq, macro_dq, events_dq, stockouts_dq, audit_summary
    )
    print(f"  Stage 1 completed in {time.time() - t0:.2f}s! Cleaned {len(train_clean):,} train transactions.")

    if stage == "clean":
        return

    # ---------------------------------------------------------
    # STAGE 2: ANOMALY DETECTION & UNCONSTRAINED DEMAND
    # ---------------------------------------------------------
    print("\n>>> [STAGE 2/6] Reconstructing Unconstrained Demand & Running Anomaly Detection...")
    t0 = time.time()

    unconstrained_df = reconstruct_unconstrained_demand(train_clean, stockouts_clean)
    anomaly_register = detect_anomalies(unconstrained_df)
    cycle_df, switch_df, lost_df = analyze_customer_cycles(unconstrained_df)

    print(f"  Stage 2 completed in {time.time() - t0:.2f}s! Flagged {len(anomaly_register):,} anomalies; analyzed {len(cycle_df):,} ordering cycles.")

    if stage == "anomaly":
        return

    # ---------------------------------------------------------
    # STAGE 3: FEATURE ENGINEERING & DRIVER ANALYSIS
    # ---------------------------------------------------------
    print("\n>>> [STAGE 3/6] Building Feature Store, Macro Drivers & EV Powertrain Dynamics...")
    t0 = time.time()

    train_features, test_features = build_feature_store(
        unconstrained_df, raw_test, sku_master_clean, macro_clean, events_clean
    )
    full_panel = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "full_panel_features.parquet")

    print(f"  Stage 3 completed in {time.time() - t0:.2f}s! Feature matrix shape: {train_features.shape}")

    if stage == "features":
        return

    # ---------------------------------------------------------
    # STAGE 4: ROLLING-ORIGIN BACKTESTING & MODEL SELECTION
    # ---------------------------------------------------------
    print("\n>>> [STAGE 4/6] Executing Rolling-Origin Backtesting (Origins: 2023-10, 2024-10, 2025-03)...")
    t0 = time.time()

    backtest_metrics, backtest_preds = run_rolling_origin_backtest(
        full_panel, sku_master_clean, target_col="Units_Sold_Unconstrained"
    )
    family_selection = select_best_model_per_family(backtest_preds)

    print(f"  Stage 4 completed in {time.time() - t0:.2f}s!")
    print("  Backtest Summary (WAPE % by Model and Origin):")
    if not backtest_metrics.empty:
        print(backtest_metrics.to_string(index=False))

    if stage == "backtest":
        return

    # ---------------------------------------------------------
    # STAGE 5: FINAL MULTI-QUANTILE FORECASTING & RECONCILIATION
    # ---------------------------------------------------------
    print("\n>>> [STAGE 5/6] Training Final Multi-Quantile LightGBM Engine (P10, P50, P90)...")
    t0 = time.time()

    lgb_preds, lgb_meta = train_and_forecast_lgb(
        train_features, test_features, target_col="Units_Sold_Unconstrained"
    )

    # Attach predictions to test features for lifecycle constraints
    test_features["P10"] = lgb_preds["P10"].values
    test_features["P50"] = lgb_preds["P50"].values
    test_features["P90"] = lgb_preds["P90"].values

    # Apply EOL boundaries and cold-start transfer learning
    test_features = apply_lifecycle_and_cold_start(
        test_features, sku_master_clean, train_features
    )
    lgb_preds["P10"] = test_features["P10"].values
    lgb_preds["P50"] = test_features["P50"].values
    lgb_preds["P90"] = test_features["P90"].values

    # Hierarchical Reconciliation & Submission Generation
    submission_df, intervals_df, recon_check = reconcile_and_generate_submission(
        test_features, lgb_preds, raw_test, sku_master_clean
    )

    print(f"  Stage 5 completed in {time.time() - t0:.2f}s!")
    print(f"  Generated competition submission: {len(submission_df):,} rows.")
    print(f"  Total 19-Month Forecast Volume: {submission_df['Units_Sold_Forecast'].sum():,} units.")
    print(f"  Hierarchy Discrepancy: {recon_check['Discrepancy'].abs().sum()} (0 = exact addition)")

    # ---------------------------------------------------------
    # STAGE 6: GOVERNANCE & EXECUTIVE REPORTS
    # ---------------------------------------------------------
    print("\n>>> [STAGE 6/6] Generating Model Card and Strategic Executive Summary...")
    t0 = time.time()

    card_path = generate_model_card(backtest_metrics, family_selection, lgb_meta)
    summary_path = generate_executive_summary(unconstrained_df, submission_df)

    print(f"  Stage 6 completed in {time.time() - t0:.2f}s!")
    print(f"  - Model Card: {card_path}")
    print(f"  - Executive Summary: {summary_path}")

    total_elapsed = time.time() - t0_master
    print("\n" + "=" * 80)
    print(f"  MASTER PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS!")
    print(f"  - Cleaned Datasets: data/cleaned/")
    print(f"  - Feature Store:    data/features/")
    print(f"  - Submission File:  outputs/submission.csv & submission.csv ({len(submission_df):,} rows)")
    print(f"  - Reports & Audits: reports/")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Celeris Automobili Demand Forecasting Master Pipeline")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "clean", "anomaly", "features", "backtest", "forecast"],
                        help="Execution stage to run (default: all)")
    args = parser.parse_args()
    run_pipeline(args.stage)

if __name__ == "__main__":
    main()
