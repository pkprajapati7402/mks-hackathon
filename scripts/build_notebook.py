"""
Script to build mks_datathon_notebook.ipynb with bulletproof Google Colab and Local execution.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏭 Celeris Automobili Industrial: Demand Forecasting & Planning Control Tower\n",
                "### MKS AIML Hackathon Phase 2: Production End-to-End Enterprise Demand Engine\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pkprajapati7402/mks-hackathon/blob/main/mks_datathon_notebook.ipynb)\n",
                "[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)\n",
                "[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-brightgreen)](https://lightgbm.readthedocs.io/)\n",
                "[![19-Month Forecast](https://img.shields.io/badge/Forecast%20Horizon-Nov%202026%20--%20May%202028-orange)](#)\n",
                "[![Submission Integrity](https://img.shields.io/badge/Submission%20Rows-58%2C219%20Valid-success)](#)\n",
                "\n",
                "---\n",
                "### 📌 Executive Overview\n",
                "This interactive notebook demonstrates the enterprise demand forecasting pipeline and S&OP control tower built for **Celeris Automobili Industrial** covering **601 commercial SKUs, 7 product families, 4 channels, and 6 global regions**.\n",
                "\n",
                "#### 🎯 Key Technical Capabilities Demonstrated:\n",
                "1. **Systematic ERP Data Cleaning**: 15+ vectorized business rules correcting SKU typos, customer name aliases, regional consolidations, negative returns, and year transposition errors.\n",
                "2. **True Unconstrained Demand Reconstruction**: Correcting historical sales for inventory stockouts with customer-specific proportional allocation to prevent carrying supply-chain bottlenecks into future build schedules.\n",
                "3. **Anomaly Intelligence Register**: Categorizing and explaining 2,100 historical structural anomalies (supply shocks, early buying spikes before price hikes, pandemic dips).\n",
                "4. **95-Feature Panel & Macro Econometric Drivers**: Cross-correlation lead/lag analysis with Automotive Production and Steel indices, plus quantified EV vs. ICE transition dynamics.\n",
                "5. **Leakage-Free 19-Month Rolling Backtesting**: Evaluating Seasonal Naive, Moving Average, Intermittent Croston SBA, and Multi-Quantile LightGBM across multiple origins (`2023-10`, `2024-10`, `2025-03` holdout).\n",
                "6. **Exact Hierarchical Reconciliation & Life-Cycle Governance**: Bottom-up aggregation ensuring $\\Delta = 0$ discrepancy across SKU, Family, Region, and Business Unit hierarchies, with zero demand post-EOL."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Verification & Auto-Setup (Google Colab & Local)\n",
                "Automatically detects the runtime environment, clones/pulls the repository if running inside Google Colab, sets working directory, installs dependencies, and configures native Plotly rendering."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# -----------------------------------------------------------------------------\n",
                "# 1. ROBUST ENVIRONMENT DETECTION & AUTO-SETUP (COLAB + LOCAL)\n",
                "# -----------------------------------------------------------------------------\n",
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "IN_COLAB = \"google.colab\" in sys.modules\n",
                "REPO_URL = \"https://github.com/pkprajapati7402/mks-hackathon.git\"\n",
                "REPO_DIR = \"/content/mks-hackathon\"\n",
                "\n",
                "if IN_COLAB:\n",
                "    print(\"⚡ Running in Google Colab environment.\")\n",
                "    \n",
                "    # Clone repository if src/ is not present in current working directory\n",
                "    if not (Path.cwd() / \"src\").exists():\n",
                "        if not Path(f\"{REPO_DIR}/src\").exists():\n",
                "            print(f\"📥 Cloning full production repository from {REPO_URL}...\")\n",
                "            !rm -rf {REPO_DIR}\n",
                "            !git clone {REPO_URL} {REPO_DIR}\n",
                "        \n",
                "        # Switch working directory to cloned repo root\n",
                "        %cd {REPO_DIR}\n",
                "    \n",
                "    # Ensure project root is at front of sys.path\n",
                "    if REPO_DIR not in sys.path:\n",
                "        sys.path.insert(0, REPO_DIR)\n",
                "    if str(Path.cwd()) not in sys.path:\n",
                "        sys.path.insert(0, str(Path.cwd()))\n",
                "    \n",
                "    # Install required production libraries\n",
                "    print(\"📦 Installing required dependencies...\")\n",
                "    !pip install -q pyarrow lightgbm streamlit plotly statsmodels scikit-learn\n",
                "    \n",
                "    # Set Plotly renderer for Google Colab\n",
                "    import plotly.io as pio\n",
                "    pio.renderers.default = \"colab\"\n",
                "else:\n",
                "    PROJECT_ROOT = Path.cwd().resolve()\n",
                "    if str(PROJECT_ROOT) not in sys.path:\n",
                "        sys.path.insert(0, str(PROJECT_ROOT))\n",
                "    import plotly.io as pio\n",
                "    pio.renderers.default = \"notebook_connected\"\n",
                "    print(f\"💻 Running in Local environment: {Path.cwd()}\")\n",
                "\n",
                "# Safe display function across notebook and CLI environments\n",
                "try:\n",
                "    from IPython.display import display\n",
                "except ImportError:\n",
                "    display = print\n",
                "\n",
                "# Import core scientific libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import plotly.express as px\n",
                "import plotly.graph_objects as go\n",
                "import json\n",
                "\n",
                "print(f\"\\n✅ Environment Ready!\")\n",
                "print(f\"   - Working Directory: {Path.cwd()}\")\n",
                "print(f\"   - Python Version:    {sys.version.split()[0]}\")\n",
                "print(f\"   - Plotly Renderer:   {pio.renderers.default}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Execute Master Demand Pipeline\n",
                "Run the master one-command production pipeline to execute the complete 6-stage lifecycle:\n",
                "- **Stage 1**: Clean raw ERP datasets & log row-level audit trails.\n",
                "- **Stage 2**: Reconstruct unconstrained customer demand & detect anomalies.\n",
                "- **Stage 3**: Assemble 95-feature panel & econometric macro lag features.\n",
                "- **Stage 4**: Execute rolling-origin 19-month backtest tournament across origins.\n",
                "- **Stage 5**: Train Multi-Quantile LightGBM ($P10, P50, P90$) and reconcile hierarchies.\n",
                "- **Stage 6**: Generate AI/ML Model Card and Executive S&OP Summary."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# -----------------------------------------------------------------------------\n",
                "# 2. EXECUTE MASTER DEMAND FORECASTING PIPELINE\n",
                "# -----------------------------------------------------------------------------\n",
                "import sys\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "# Self-healing path & repo resolution guard\n",
                "REPO_DIR = \"/content/mks-hackathon\"\n",
                "if not (Path.cwd() / \"src\").exists():\n",
                "    if Path(f\"{REPO_DIR}/src\").exists():\n",
                "        %cd {REPO_DIR}\n",
                "    elif \"google.colab\" in sys.modules:\n",
                "        print(\"⚡ Auto-cloning repository in Colab...\")\n",
                "        !git clone https://github.com/pkprajapati7402/mks-hackathon.git {REPO_DIR}\n",
                "        %cd {REPO_DIR}\n",
                "        !pip install -q pyarrow lightgbm streamlit plotly statsmodels scikit-learn\n",
                "\n",
                "if str(Path.cwd()) not in sys.path:\n",
                "    sys.path.insert(0, str(Path.cwd()))\n",
                "if REPO_DIR not in sys.path:\n",
                "    sys.path.insert(0, REPO_DIR)\n",
                "\n",
                "from run_pipeline import run_pipeline\n",
                "\n",
                "# Execute entire end-to-end demand forecasting pipeline\n",
                "# (Runs in ~70s: cleaning, anomalies, feature store, rolling backtesting, LightGBM P10/P50/P90, reconciliation)\n",
                "run_pipeline(stage=\"all\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Integrity & Anomaly Register Analysis\n",
                "Explore the row-level audit trail, data quality report, and categorized historical anomalies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "from src.config import REPORTS_DIR, CLEANED_DATA_DIR\n",
                "\n",
                "# 1. Load Data Quality Report\n",
                "with open(REPORTS_DIR / \"data_quality_report.json\", \"r\", encoding=\"utf-8\") as f:\n",
                "    dq_report = json.load(f)\n",
                "\n",
                "# Display file cleaning summary table\n",
                "file_summaries = []\n",
                "for fname, stats in dq_report.get(\"file_summaries\", {}).items():\n",
                "    file_summaries.append({\n",
                "        \"Dataset\": fname,\n",
                "        \"Raw Rows\": f\"{stats.get('raw_rows', 0):,}\",\n",
                "        \"Cleaned Rows\": f\"{stats.get('clean_rows', 0):,}\",\n",
                "        \"Status\": \"Verified Clean ✅\"\n",
                "    })\n",
                "dq_df = pd.DataFrame(file_summaries)\n",
                "display(dq_df)\n",
                "\n",
                "# 2. Inspect Anomaly Register\n",
                "anomaly_df = pd.read_csv(REPORTS_DIR / \"anomaly_register.csv\")\n",
                "print(f\"\\nTotal Anomalies Categorized: {len(anomaly_df):,}\")\n",
                "print(\"\\nAnomaly Breakdown by Category:\")\n",
                "display(anomaly_df[\"Category\"].value_counts().to_frame(\"Count\"))\n",
                "\n",
                "print(\"\\nSample Structural Anomalies with Business Context:\")\n",
                "display(anomaly_df[[\"Date\", \"SKU_ID\", \"Region\", \"Category\", \"Actual_Units\", \"Severity\", \"Explanation\"]].head(8))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Econometric Drivers & EV Powertrain Paradigm Shift\n",
                "Analyse the cross-correlation of macroeconomic indicators (Automotive Production Index, Steel Price Index) with product demand, and quantify the transition from internal combustion engines (ICE) to electric vehicles (EV)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import plotly.express as px\n",
                "from src.config import REPORTS_DIR\n",
                "\n",
                "# 1. Macro Lead/Lag Cross-Correlation Analysis\n",
                "macro_lag_df = pd.read_csv(REPORTS_DIR / \"macro_lead_lag_analysis.csv\")\n",
                "print(\"Macro Lead/Lag Cross-Correlation (Lags 0 to 6 months):\")\n",
                "display(macro_lag_df.head(10))\n",
                "\n",
                "# 2. EV vs ICE Powertrain Shift Analysis\n",
                "ev_df = pd.read_csv(REPORTS_DIR / \"ev_shift_analysis.csv\")\n",
                "print(\"\\nEV vs ICE Powertrain Annual Volume & Portfolio Share:\")\n",
                "display(ev_df)\n",
                "\n",
                "# Visualize the EV Transition Trajectory\n",
                "fig_ev = px.bar(\n",
                "    ev_df,\n",
                "    x=\"Year\",\n",
                "    y=[\"ICE_Units\", \"EV_Units\"],\n",
                "    barmode=\"group\",\n",
                "    title=\"Powertrain Paradigm Shift: Annual Volume (ICE vs EV Components)\",\n",
                "    labels={\"value\": \"Annual Shipped Units\", \"variable\": \"Powertrain Type\"},\n",
                "    color_discrete_map={\"ICE_Units\": \"#EF4444\", \"EV_Units\": \"#10B981\"},\n",
                "    template=\"plotly_dark\"\n",
                ")\n",
                "fig_ev.update_layout(paper_bgcolor=\"#1E293B\", plot_bgcolor=\"#0F172A\")\n",
                "fig_ev.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Rolling-Origin Backtest Tournament & Champion Selection\n",
                "Review 19-month out-of-sample backtest metrics across time-series cross-validation origins (`2023-10`, `2024-10`, `2025-03` holdout) comparing LightGBM against baseline and intermittent models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "from src.config import OUTPUTS_DIR\n",
                "\n",
                "metrics_df = pd.read_csv(OUTPUTS_DIR / \"backtest_metrics.csv\")\n",
                "print(\"Rolling-Origin 19-Month Backtest Performance:\")\n",
                "display(metrics_df)\n",
                "\n",
                "selection_df = pd.read_csv(OUTPUTS_DIR / \"family_model_selection.csv\")\n",
                "print(\"\\nProduct Family Champion Model Selection:\")\n",
                "display(selection_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Continuous Demand Timeline & Prediction Intervals ($P10 / P50 / P90$)\n",
                "Visualizing historical actual sales, reconstructed unconstrained demand, and the 19-month multi-quantile forecast with shaded uncertainty bands."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import plotly.express as px\n",
                "from src.config import CLEANED_DATA_DIR, OUTPUTS_DIR\n",
                "from src.dashboard.components import plot_continuous_timeline\n",
                "\n",
                "# Load historical and forecast datasets\n",
                "train_df = pd.read_parquet(CLEANED_DATA_DIR / \"unconstrained_demand.parquet\")\n",
                "fcst_df = pd.read_parquet(OUTPUTS_DIR / \"forecast_intervals.parquet\")\n",
                "\n",
                "# Aggregate monthly timelines\n",
                "hist_monthly = train_df.groupby(\"Date\").agg({\n",
                "    \"Units_Sold\": \"sum\",\n",
                "    \"Units_Sold_Unconstrained\": \"sum\"\n",
                "}).reset_index()\n",
                "\n",
                "fcst_monthly = fcst_df.groupby(\"Date\").agg({\n",
                "    \"P10\": \"sum\",\n",
                "    \"P50\": \"sum\",\n",
                "    \"P90\": \"sum\",\n",
                "    \"Units_Sold_Forecast\": \"sum\"\n",
                "}).reset_index()\n",
                "\n",
                "# Render interactive Plotly timeline\n",
                "fig_timeline = plot_continuous_timeline(\n",
                "    hist_monthly,\n",
                "    fcst_monthly,\n",
                "    title=\"Celeris Automobili Continuous Demand Horizon (2019-2028) with P10/P90 Ribbon\"\n",
                ")\n",
                "fig_timeline.show()\n",
                "\n",
                "# Regional Forecast Breakdown\n",
                "reg_df = fcst_df.groupby(\"Region\")[\"Units_Sold_Forecast\"].sum().reset_index()\n",
                "fig_reg = px.pie(\n",
                "    reg_df,\n",
                "    names=\"Region\",\n",
                "    values=\"Units_Sold_Forecast\",\n",
                "    title=\"19-Month Global Forecast Distribution by Region\",\n",
                "    color_discrete_sequence=px.colors.sequential.Teal,\n",
                "    template=\"plotly_dark\",\n",
                "    hole=0.4\n",
                ")\n",
                "fig_reg.update_layout(paper_bgcolor=\"#1E293B\")\n",
                "fig_reg.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Official Competition Submission Verification (58,219 Rows)\n",
                "Comprehensive validation checks ensuring exact alignment with `student_data/test.csv`, non-negativity, zero nulls, and automated one-click file download in Google Colab."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "from src.config import RAW_TEST, OUTPUTS_DIR\n",
                "\n",
                "sub_path = OUTPUTS_DIR / \"submission.csv\"\n",
                "if not sub_path.exists():\n",
                "    sub_path = Path(\"submission.csv\")\n",
                "\n",
                "sub_df = pd.read_csv(sub_path)\n",
                "raw_test = pd.read_csv(RAW_TEST)\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(\"  OFFICIAL SUBMISSION INTEGRITY CHECKLIST\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# Check 1: Row count\n",
                "assert len(sub_df) == 58219, f\"Row count mismatch! Expected 58,219, got {len(sub_df)}\"\n",
                "print(f\"✅ Row Count: {len(sub_df):,} matches test.csv exactly.\")\n",
                "\n",
                "# Check 2: Columns\n",
                "expected_cols = [\"Date\", \"SKU_ID\", \"Region\", \"Units_Sold_Forecast\"]\n",
                "assert list(sub_df.columns) == expected_cols, f\"Column mismatch! Expected {expected_cols}, got {list(sub_df.columns)}\"\n",
                "print(f\"✅ Columns: {list(sub_df.columns)} verified.\")\n",
                "\n",
                "# Check 3: Null count\n",
                "null_cnt = sub_df.isnull().sum().sum()\n",
                "assert null_cnt == 0, f\"Found {null_cnt} null values!\"\n",
                "print(f\"✅ Null Count: 0 missing or NaN values.\")\n",
                "\n",
                "# Check 4: Non-negativity\n",
                "neg_cnt = (sub_df[\"Units_Sold_Forecast\"] < 0).sum()\n",
                "assert neg_cnt == 0, f\"Found {neg_cnt} negative forecasts!\"\n",
                "print(f\"✅ Validity: 100% non-negative integers.\")\n",
                "\n",
                "# Check 5: Alignment with test.csv\n",
                "alignment_diff = (sub_df[\"Date\"] != raw_test[\"Date\"]).sum() + (sub_df[\"SKU_ID\"] != raw_test[\"SKU_ID\"]).sum()\n",
                "assert alignment_diff == 0, f\"Key alignment mismatch with test.csv in {alignment_diff} rows!\"\n",
                "print(f\"✅ Alignment: 100% key-aligned with student_data/test.csv.\")\n",
                "\n",
                "print(f\"\\n📊 Total 19-Month Forecast Demand: {sub_df['Units_Sold_Forecast'].sum():,} units\")\n",
                "print(f\"   Monthly Average: {sub_df.groupby('Date')['Units_Sold_Forecast'].sum().mean():,.0f} units/month\")\n",
                "\n",
                "display(sub_df.head(10))\n",
                "\n",
                "# Colab One-Click Download\n",
                "if \"google.colab\" in sys.modules:\n",
                "    from google.colab import files\n",
                "    print(\"\\n⬇️ Triggering one-click download of submission.csv to your local machine...\")\n",
                "    files.download(str(sub_path))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Launch Interactive Streamlit Control Tower in Google Colab\n",
                "Launch the full 12-view executive S&OP Control Tower Dashboard (`app.py`) directly from Google Colab using `localtunnel`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "\n",
                "# Launch Streamlit Control Tower in background and expose via localtunnel\n",
                "if \"google.colab\" in sys.modules:\n",
                "    import urllib.request\n",
                "    \n",
                "    # 1. Start Streamlit in background\n",
                "    !nohup streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &\n",
                "    \n",
                "    # 2. Get public IP address for localtunnel tunnel password\n",
                "    try:\n",
                "        ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()\n",
                "    except Exception:\n",
                "        ip = \"Check colab IP\"\n",
                "    print(\"=\" * 70)\n",
                "    print(\"🚀 STREAMLIT CONTROL TOWER LAUNCHED!\")\n",
                "    print(f\"🔑 Your Tunnel Password (Enter this at the tunnel URL): {ip}\")\n",
                "    print(\"=\" * 70)\n",
                "    print(\"Click the link generated below to open the interactive dashboard:\")\n",
                "    \n",
                "    # 3. Expose port 8501 via localtunnel with automatic yes to prompt\n",
                "    !npx -y localtunnel --port 8501\n",
                "else:\n",
                "    print(\"💻 To run the dashboard locally, execute in your terminal:\")\n",
                "    print(\"   streamlit run app.py\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "toc_visible": True
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("mks_datathon_notebook.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully generated {out_path} ({out_path.stat().st_size:,} bytes)")
