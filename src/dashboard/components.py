"""
Reusable UI Components, Plotly Visualizations, and Theme Styling for the Control Tower.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

try:
    import streamlit as st
except ImportError:
    st = None

# Custom CSS for World-Class Executive Supply Chain Dashboard
CUSTOM_CSS = """
<style>
    /* Main Layout & Dark Elegance */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Banner */
    .control-tower-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .control-tower-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    
    .control-tower-subtitle {
        font-size: 14px;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #00E5FF;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #64748B;
    }

    /* Status Badges */
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-info {
        background-color: rgba(0, 229, 255, 0.15);
        color: #00E5FF;
        border: 1px solid rgba(0, 229, 255, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
"""

def render_kpi_card(label: str, value: str, subtext: str = ""):
    """Renders a polished dark-theme KPI metric card."""
    if st is not None:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        print(f"[{label}] {value} ({subtext})")

def plot_continuous_timeline(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    title: str = "Continuous Demand History and 19-Month Forecast Timeline",
    show_intervals: bool = True
) -> go.Figure:
    """
    Renders history (2019-01 -> 2026-10) and forecast (2026-11 -> 2028-05) on one seamless timeline
    with a vertical marker separating history from forecast, showing unconstrained demand and P10-P90 bands.
    """
    fig = go.Figure()

    # 1. Historical Actual Shipped Units
    fig.add_trace(go.Scatter(
        x=history_df["Date"],
        y=history_df["Units_Sold"],
        mode="lines",
        name="Historical Shipped",
        line=dict(color="#64748B", width=1.8),
        hovertemplate="<b>%{x}</b><br>Shipped: %{y:,.0f} units<extra></extra>"
    ))

    # 2. Historical Unconstrained Demand
    if "Units_Sold_Unconstrained" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=history_df["Date"],
            y=history_df["Units_Sold_Unconstrained"],
            mode="lines",
            name="Unconstrained Demand (True)",
            line=dict(color="#00E5FF", width=2.5),
            hovertemplate="<b>%{x}</b><br>Unconstrained: %{y:,.0f} units<extra></extra>"
        ))

    # 3. Forecast P10 & P90 Shaded Band
    if show_intervals and "P10" in forecast_df.columns and "P90" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["P90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["P10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 229, 255, 0.15)",
            name="Prediction Interval (P10 - P90)",
            hoverinfo="skip"
        ))

    # 4. Forecast Primary P50 Line
    fcst_col = "P50" if "P50" in forecast_df.columns else "Units_Sold_Forecast"
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df[fcst_col],
        mode="lines+markers",
        name="Forecast (P50 Median)",
        line=dict(color="#F59E0B", width=2.8, dash="solid"),
        marker=dict(size=5, color="#F59E0B"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:,.0f} units<extra></extra>"
    ))

    # 5. Add Cutoff Line separating History from Forecast
    fig.add_vline(
        x="2026-10",
        line_width=2,
        line_dash="dash",
        line_color="#94A3B8",
        annotation_text="Forecast Window (Nov 2026)",
        annotation_position="top right",
        annotation_font=dict(color="#F8FAFC", size=11)
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F8FAFC", size=16)),
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1E293B", showgrid=True),
        yaxis=dict(gridcolor="#1E293B", showgrid=True, title="Demand Volume (Units)")
    )

    return fig
