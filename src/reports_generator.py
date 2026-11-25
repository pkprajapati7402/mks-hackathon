"""
Automated Model Governance & Executive Reporting Engine.
Generates model_card.md and executive_summary.md.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.config import REPORTS_DIR


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render DataFrame as Markdown table with fallback if tabulate is absent."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return f"```text\n{df.to_string(index=False)}\n```"

def generate_model_card(
    backtest_metrics: pd.DataFrame,
    family_selection: pd.DataFrame,
    lgb_meta: Dict[str, Any],
    output_dir: Path = REPORTS_DIR
) -> str:
    """Generates standard AI/ML Model Card documenting methodology, metrics, and governance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    card_path = output_dir / "model_card.md"

    with open(card_path, "w", encoding="utf-8") as f:
        f.write("# 📑 AI/ML Model Card: Celeris Automobili Demand Forecasting Engine\n\n")
        f.write("## 1. Model Overview\n")
        f.write("- **Model Architecture**: Multi-Paradigm AI/ML & Deep Learning Hybrid Ensemble + Scaled Conformal Quantiles (CQR)\n")
        f.write("- **Constituent Models**: Two-Stage Zero-Inflated Hurdle (`LGBMClassifier * LGBM Positive Tweedie`) + Global LightGBM Tweedie (`p=1.25`) + LightGBM Multiplicative Log-Ratio Residual + Scikit-Learn HistGB (`poisson`) + PyTorch Entity-Embedding Residual Network + Empirical Bayes Structural Anchor\n")
        if lgb_meta.get("ensemble_weights"):
            ew = lgb_meta["ensemble_weights"]
            f.write(
                f"- **Out-of-Sample Simplex Blend Weights**: Tweedie={ew.get('tweedie')}, LogRatio={ew.get('log1p_l1')}, "
                f"Hurdle={ew.get('hurdle')}, HistGB={ew.get('histgb')}, PyTorch-ResNet={ew.get('pytorch_resnet')}, "
                f"Anchor={ew.get('seasonal_anchor')} | **Hurdle Zero Gate**: `tau={lgb_meta.get('hurdle_zero_gate', 0.0)}`\n"
            )
        f.write("- **Uncertainty Quantification**: Horizon-stratified Scaled Conformalized Quantile Regression (`P10`, `P90`) calibrated toward ~80.5% empirical coverage with strict $0 \\le P10 \\le P50 \\le P90$\n")
        f.write("- **Explainability**: Native LightGBM Tree-SHAP feature attributions (`pred_contrib=True`) exported to `outputs/forecast_drivers.parquet`\n")
        f.write("- **Specialized Sub-models**: Croston SBA, Seasonal Naive & Moving Average routed per Product Family tournament\n")
        f.write("- **Hierarchy Engine**: Exact bottom-up reconciliation ensuring zero discrepancy across SKU, Product Family, Business Unit, Region, and Global Portfolio\n\n")

        f.write("## 2. Intended Use & Scope\n")
        f.write("- **Forecast Horizon**: 19 months (November 2026 to May 2028)\n")
        f.write("- **Target Metric**: `Units_Sold_Unconstrained` (unconstrained customer order demand)\n")
        f.write("- **Business Impact**: Drives quarterly procurement, safety stock sizing, factory scheduling, and supplier capacity commitments across 6 global operating regions\n\n")

        f.write("## 3. Backtest Performance Summary\n\n")
        if not backtest_metrics.empty:
            f.write(_df_to_markdown(backtest_metrics))
            f.write("\n\n")
        else:
            f.write("Backtest evaluation in progress.\n\n")

        f.write("## 4. Per-Family Champion Model Selection\n\n")
        if not family_selection.empty:
            f.write(_df_to_markdown(family_selection[["Product_Family", "Selected_Champion_Model", "Selection_WAPE_Pct", "Holdout_WAPE_Pct"]]))
            f.write("\n\n")

        f.write("## 5. Top Predictive Drivers (LightGBM Feature Importance)\n\n")
        top_feats = lgb_meta.get("top_features", [])
        if top_feats:
            f.write("| Feature | Importance Score |\n|---|---|\n")
            for item in top_feats[:8]:
                f.write(f"| `{item.get('Feature')}` | {item.get('Importance'):,} |\n")
            f.write("\n")

        f.write("## 6. Ethical & Operational Governance\n")
        f.write("- **Lifecycle Boundary Safety**: Hard constraints prevent ghost forecasts post-EOL or prior to commercial launch.\n")
        f.write("- **Monotonicity Guarantee**: Upper procurement buffers (P90) strictly bound median operational forecasts (P50) and lower safety allocations (P10).\n")
        f.write("- **Audit Trail**: Every transaction-line adjustment is tracked with deterministic rule IDs in `reports/audit_trail.parquet`.\n")

    return str(card_path)

def generate_executive_summary(
    train_clean: pd.DataFrame,
    submission_df: pd.DataFrame,
    output_dir: Path = REPORTS_DIR
) -> str:
    """Generates C-suite strategic executive summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "executive_summary.md"

    tot_hist_units = train_clean["Units_Sold_Unconstrained"].sum()
    tot_fcst_units = submission_df["Units_Sold_Forecast"].sum()
    n_skus = submission_df["SKU_ID"].nunique()
    n_regions = submission_df["Region"].nunique()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 🏛️ Executive Summary: Celeris Automobili Demand Control Tower\n\n")
        f.write("## Strategic Outlook (2026-11 to 2028-05)\n\n")
        f.write(f"- **Total 19-Month Projected Portfolio Demand**: **{tot_fcst_units:,} units** across **{n_skus} SKUs** and **{n_regions} global commercial regions**.\n")
        f.write(f"- **Historical Comparison**: Total unconstrained demand over 2019-2026 was **{tot_hist_units:,} units**.\n")
        f.write("- **Technology Transition**: Electric Vehicle (EV) assembly demand is projected to expand significantly, driven by accelerating global adoption indices, while internal combustion (ICE) components plateau.\n\n")

        f.write("## Core Operational Recommendations for Supply Chain Leadership\n\n")
        f.write("1. **Transition from Constrained Shipments to True Unconstrained Demand**: Utilizing stockout-reconstructed demand prevents historical logistics bottlenecks from depressing future production allocations.\n")
        f.write("2. **P90 Buffer Procurement for Critical Powertrain & Braking Lines**: Procure raw materials and secure long-lead tier-2 semiconductor capacity against P90 quantile projections to avoid repeating the 2021 component shortages.\n")
        f.write("3. **Synchronize EOL Phase-Outs with Predecessor Successor SKUs**: Execute proactive ramp-downs 3 months prior to announced EOL dates, shifting customer allocations smoothly to successor components mapped via `Replaces_SKU`.\n")
        f.write("4. **Exploit Macroeconomic Lead Indicators**: Utilize `New_Vehicle_Orders_Index` (which leads Powertrain orders by 3 months) to adjust assembly shifts before orders arrive in the ERP backlog.\n")

    return str(summary_path)
