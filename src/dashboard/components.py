"""
Reusable UI Components, Plotly Visualizations, and Theme Styling for the Control Tower.
"""

import html as _html
import inspect
import textwrap
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

try:
    import streamlit as st
except ImportError:
    st = None

# Compact CSS with no blank lines or leading indentation so HTML/Markdown parsers never emit raw text
CUSTOM_CSS = textwrap.dedent("""
<style>
.stApp {
    background-color: #0B0F19;
    color: #F3F4F6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
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
.kpi-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
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
""").strip()


def render_html_block(html_body: str, allow_style: bool = False) -> None:
    """
    Renders styled HTML/CSS without relying on Markdown rehype-raw dynamic chunks.
    Uses st.html() natively when available, falling back to zero-indent st.markdown().
    """
    compact = "\n".join(line.strip() for line in html_body.strip().splitlines() if line.strip())
    if st is None:
        return
    if hasattr(st, "html"):
        if allow_style and "unsafe_allow_javascript" in inspect.signature(st.html).parameters:
            st.html(compact, unsafe_allow_javascript=True)
            return
        if not allow_style:
            st.html(compact)
            return
    st.markdown(compact, unsafe_allow_html=True)


def render_chart(fig: go.Figure) -> None:
    """Renders a Plotly chart across Streamlit versions without deprecation warnings."""
    if st is None:
        return
    params = inspect.signature(st.plotly_chart).parameters
    if "width" in params:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def render_table(df: pd.DataFrame) -> None:
    """Renders a DataFrame across Streamlit versions without deprecation warnings."""
    if st is None:
        return
    params = inspect.signature(st.dataframe).parameters
    if "width" in params:
        st.dataframe(df, width="stretch")
    else:
        st.dataframe(df, use_container_width=True)


def render_kpi_card(label: str, value: str, subtext: str = ""):
    """Renders a dark-theme KPI metric card cleanly via st.html without CommonMark indentation issues."""
    if st is not None:
        safe_label = _html.escape(str(label))
        safe_value = _html.escape(str(value))
        safe_subtext = _html.escape(str(subtext))
        card_html = (
            '<div class="kpi-card" style="background:#1E293B;border:1px solid #334155;'
            'border-radius:10px;padding:18px 20px;box-shadow:0 2px 10px rgba(0,0,0,0.25);">'
            f'<div class="kpi-label" style="font-size:12px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.8px;color:#94A3B8;margin-bottom:6px;">{safe_label}</div>'
            f'<div class="kpi-value" style="font-size:26px;font-weight:700;color:#F8FAFC;'
            f'margin-bottom:4px;">{safe_value}</div>'
            f'<div class="kpi-subtext" style="font-size:12px;color:#64748B;">{safe_subtext}</div>'
            "</div>"
        )
        render_html_block(card_html)
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

    # 5. Cutoff line separating History from Forecast (shape+annotation works across Plotly 5.x-7.x)
    fig.add_shape(
        type="line",
        x0="2026-10",
        x1="2026-10",
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#94A3B8", width=2, dash="dash")
    )
    fig.add_annotation(
        x="2026-10",
        y=1.0,
        yref="paper",
        text="Forecast Window (Nov 2026)",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(color="#F8FAFC", size=11)
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
