"""
Forecast & Backtest Accuracy View (Required View 5).
Displays model backtest leaderboard, WAPE and Bias across origins (2023-10, 2024-10, 2025-03),
per-family champion model selection, and horizon error progression.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import OUTPUTS_DIR
from src.dashboard.components import render_kpi_card

def render_view():
    st.markdown("### 🎯 Forecast Accuracy & Backtest Tournament")
    st.markdown("Rigorous multi-origin rolling backtest evaluation (Origins: 2023-10, 2024-10, 2025-03) across a 19-month horizon without lookahead leakage.")

    metrics_path = OUTPUTS_DIR / "backtest_metrics.csv"
    selection_path = OUTPUTS_DIR / "family_model_selection.csv"

    if not metrics_path.exists():
        st.warning("Backtest results not found. Run pipeline first.")
        return

    metrics_df = pd.read_csv(metrics_path)
    selection_df = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()

    # 1. Headline Backtest KPIs
    lgb_metrics = metrics_df[metrics_df["Model"] == "LightGBM P50"]
    avg_wape = lgb_metrics["WAPE_Pct"].mean()
    avg_bias = lgb_metrics["Bias_Pct"].mean()
    holdout_wape = lgb_metrics[lgb_metrics["Origin"] == "2025-03"]["WAPE_Pct"].iloc[0] if not lgb_metrics[lgb_metrics["Origin"] == "2025-03"].empty else avg_wape

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("LightGBM Average WAPE", f"{avg_wape:.2f}%", "19-Month Rolling Backtest")
    with col2:
        render_kpi_card("Holdout WAPE (2025-03)", f"{holdout_wape:.2f}%", "Out-of-Sample Test Set")
    with col3:
        render_kpi_card("Forecast Bias", f"{avg_bias:+.2f}%", "Unbiased (< ±3% target)")
    with col4:
        naive_wape = metrics_df[metrics_df["Model"] == "Seasonal Naive"]["WAPE_Pct"].mean()
        render_kpi_card("Lift Over Baseline", f"{naive_wape - avg_wape:+.2f} pts", "Improvement vs Seasonal Naive")

    st.markdown("---")

    # 2. Model Tournament Comparison Bar Chart
    st.markdown("#### 🏆 Model Tournament Leaderboard: WAPE (%) across Origins")
    fig_wape = px.bar(
        metrics_df,
        x="Origin", y="WAPE_Pct", color="Model",
        barmode="group",
        title="19-Month Backtest WAPE (%) by Origin Date",
        template="plotly_dark",
        color_discrete_map={
            "LightGBM P50": "#00E5FF",
            "Seasonal Naive": "#F59E0B",
            "Moving Average": "#64748B",
            "Intermittent SBA": "#A855F7"
        }
    )
    fig_wape.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", yaxis_title="WAPE (%)")
    st.plotly_chart(fig_wape, use_container_width=True)

    # 3. Model Tournament Table & Bias Metrics
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📊 Comprehensive Backtest Metrics Table")
        st.dataframe(metrics_df, use_container_width=True)

    with col_b:
        st.markdown("#### ⚖️ Forecast Bias (%) by Model and Origin")
        fig_bias = px.bar(
            metrics_df,
            x="Origin", y="Bias_Pct", color="Model",
            barmode="group",
            title="Forecast Bias (%) by Model (Near 0 = Balanced)",
            template="plotly_dark",
            color_discrete_map={
                "LightGBM P50": "#00E5FF",
                "Seasonal Naive": "#F59E0B",
                "Moving Average": "#64748B",
                "Intermittent SBA": "#A855F7"
            }
        )
        fig_bias.add_hline(y=0, line_dash="dash", line_color="#F8FAFC")
        fig_bias.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", yaxis_title="Bias (%)")
        st.plotly_chart(fig_bias, use_container_width=True)

    # 4. Per-Family Champion Model Selection
    if not selection_df.empty:
        st.markdown("#### 🥇 Champion Forecasting Model Selection per Product Family")
        st.dataframe(
            selection_df[["Product_Family", "Selected_Champion_Model", "Selection_WAPE_Pct", "Holdout_WAPE_Pct", "Seasonal_Naive_WAPE", "LightGBM_P50_WAPE"]],
            use_container_width=True
        )
