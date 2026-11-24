"""
Celeris Automobili Industrial: Demand Forecasting & Planning Control Tower.
Production Streamlit Application implementing all 5 Required Views and all 7 Optional Views.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Celeris Automobili | Demand Control Tower",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import (
    CLEANED_DATA_DIR, OUTPUTS_DIR, RAW_TEST, REPORTS_DIR
)
from src.dashboard.components import CUSTOM_CSS, render_html_block
from src.dashboard.views import (
    executive, data_health, demand_explorer, anomaly_monitor,
    forecast_accuracy, macro_heatmap, what_if, procurement,
    lifecycle, stockouts, exceptions, export_data
)

# Inject Dark Enterprise CSS Styling safely via st.html / fallback
render_html_block(CUSTOM_CSS, allow_style=True)

# Cached Data Loader
@st.cache_data(show_spinner=False)
def load_control_tower_data():
    train_path = CLEANED_DATA_DIR / "unconstrained_demand.parquet"
    if not train_path.exists():
        train_path = CLEANED_DATA_DIR / "clean_train.parquet"
    
    train_df = pd.read_parquet(train_path) if train_path.exists() else pd.DataFrame()
    
    intervals_path = OUTPUTS_DIR / "forecast_intervals.parquet"
    if intervals_path.exists():
        forecast_df = pd.read_parquet(intervals_path)
    else:
        sub_path = OUTPUTS_DIR / "submission.csv"
        forecast_df = pd.read_csv(sub_path) if sub_path.exists() else pd.DataFrame()
        if "P50" not in forecast_df.columns and "Units_Sold_Forecast" in forecast_df.columns:
            forecast_df["P50"] = forecast_df["Units_Sold_Forecast"]
            forecast_df["P10"] = (forecast_df["P50"] * 0.85).round()
            forecast_df["P90"] = (forecast_df["P50"] * 1.15).round()

    sku_path = CLEANED_DATA_DIR / "sku_master.csv"
    sku_master = pd.read_csv(sku_path) if sku_path.exists() else pd.DataFrame()

    test_raw = pd.read_csv(RAW_TEST, low_memory=False) if RAW_TEST.exists() else pd.DataFrame()

    # Defensively enrich forecast_df with hierarchy, customer/channel, and powertrain columns if missing
    if not forecast_df.empty and not test_raw.empty:
        enrich_cols = [c for c in ["Product_Family", "Business_Unit", "Customer_Name", "Sales_Channel"]
                       if c in test_raw.columns and c not in forecast_df.columns]
        if enrich_cols:
            forecast_df = forecast_df.merge(
                test_raw[["Date", "SKU_ID", "Region"] + enrich_cols].drop_duplicates(subset=["Date", "SKU_ID", "Region"]),
                on=["Date", "SKU_ID", "Region"],
                how="left"
            )
    if not forecast_df.empty and not sku_master.empty and "Powertrain_Application" not in forecast_df.columns:
        sku_to_pt = dict(zip(sku_master["SKU_ID"], sku_master["Powertrain_Application"]))
        forecast_df["Powertrain_Application"] = forecast_df["SKU_ID"].map(sku_to_pt).fillna("Universal")

    return train_df, forecast_df, sku_master, test_raw

def main():
    # Load Data
    train_df, forecast_df, sku_master, test_raw = load_control_tower_data()

    if train_df.empty or forecast_df.empty:
        st.error("Pipeline outputs not detected! Please run `python run_pipeline.py` first to generate cleaned data and forecasts.")
        return

    # Top Brand Header Banner (rendered cleanly via st.html without CommonMark indentation)
    header_html = (
        '<div class="control-tower-header" style="background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);'
        'border:1px solid #334155;border-radius:12px;padding:24px 32px;margin-bottom:24px;'
        'box-shadow:0 4px 20px rgba(0,0,0,0.4);">'
        '<div class="control-tower-title" style="font-size:28px;font-weight:700;letter-spacing:-0.5px;'
        'color:#F8FAFC;margin-bottom:6px;">🏭 CELERIS AUTOMOBILI INDUSTRIAL</div>'
        '<div class="control-tower-subtitle" style="font-size:14px;color:#94A3B8;line-height:1.5;">'
        'Enterprise Demand Forecasting &amp; Operational Planning Control Tower<br>'
        'Multi-Tier Demand Engine &bull; 601 SKUs &bull; 7 Product Families &bull; 6 Global Commercial Regions'
        '</div></div>'
    )
    render_html_block(header_html)

    # Sidebar Navigation & Global Shared Filters
    with st.sidebar:
        st.markdown("### 🎛️ Navigation & Controls")
        
        selected_view = st.radio(
            "Select Control Tower View:",
            [
                "🏛️ Executive Overview",
                "🩺 Data Health & Audit",
                "🔍 Demand Explorer",
                "🚨 Anomaly Monitor",
                "🎯 Forecast Accuracy",
                "🌡️ Macro Lead/Lag Heatmap",
                "🎛️ What-If Scenario Simulator",
                "📦 Procurement & Safety Stock",
                "🔄 Lifecycle & Transition",
                "⚠️ Stockouts & Lost Sales",
                "📋 Planner Exception Center",
                "💾 Data Export & Reports"
            ]
        )

        st.markdown("---")
        st.markdown("### 🔍 Global Shared Filters")

        # 1. Product Family Filter
        families = ["All"] + sorted(train_df["Product_Family"].dropna().unique().tolist())
        selected_family = st.selectbox("Product Family:", families)

        # 2. Region Filter
        regions = ["All"] + sorted(train_df["Region"].dropna().unique().tolist())
        selected_region = st.selectbox("Sales Region:", regions)

        # 3. Channel Filter
        channels = ["All"] + sorted(train_df["Sales_Channel"].dropna().unique().tolist())
        selected_channel = st.selectbox("Sales Channel:", channels)

        # 4. Customer Filter
        customers = ["All"] + sorted(train_df["Customer_Name"].dropna().unique().tolist())
        selected_customer = st.selectbox("Tier-1 Customer:", customers)

        st.markdown("---")
        st.caption("MKS AIML Hackathon Phase 2 &bull; Production Release v1.0")

    # Apply Global Filters to Datasets
    filtered_train = train_df.copy()
    filtered_forecast = forecast_df.copy()

    if selected_family != "All":
        filtered_train = filtered_train[filtered_train["Product_Family"] == selected_family]
        if "Product_Family" in filtered_forecast.columns:
            filtered_forecast = filtered_forecast[filtered_forecast["Product_Family"] == selected_family]

    if selected_region != "All":
        filtered_train = filtered_train[filtered_train["Region"] == selected_region]
        filtered_forecast = filtered_forecast[filtered_forecast["Region"] == selected_region]

    if selected_channel != "All":
        filtered_train = filtered_train[filtered_train["Sales_Channel"] == selected_channel]
        if "Sales_Channel" in filtered_forecast.columns:
            filtered_forecast = filtered_forecast[filtered_forecast["Sales_Channel"] == selected_channel]

    if selected_customer != "All":
        filtered_train = filtered_train[filtered_train["Customer_Name"] == selected_customer]
        if "Customer_Name" in filtered_forecast.columns:
            filtered_forecast = filtered_forecast[filtered_forecast["Customer_Name"] == selected_customer]

    # Route to Selected View
    if selected_view == "🏛️ Executive Overview":
        executive.render_view(filtered_train, filtered_forecast, test_raw)
    elif selected_view == "🩺 Data Health & Audit":
        data_health.render_view()
    elif selected_view == "🔍 Demand Explorer":
        demand_explorer.render_view(filtered_train, sku_master)
    elif selected_view == "🚨 Anomaly Monitor":
        anomaly_monitor.render_view(filtered_train)
    elif selected_view == "🎯 Forecast Accuracy":
        forecast_accuracy.render_view()
    elif selected_view == "🌡️ Macro Lead/Lag Heatmap":
        macro_heatmap.render_view()
    elif selected_view == "🎛️ What-If Scenario Simulator":
        what_if.render_view(filtered_forecast)
    elif selected_view == "📦 Procurement & Safety Stock":
        procurement.render_view(filtered_forecast)
    elif selected_view == "🔄 Lifecycle & Transition":
        lifecycle.render_view(sku_master, filtered_train, filtered_forecast)
    elif selected_view == "⚠️ Stockouts & Lost Sales":
        stockouts.render_view()
    elif selected_view == "📋 Planner Exception Center":
        exceptions.render_view(filtered_forecast, sku_master)
    elif selected_view == "💾 Data Export & Reports":
        export_data.render_view()

if __name__ == "__main__":
    main()
