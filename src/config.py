"""
Central Configuration for Celeris Automobili Demand Forecasting & Planning Control Tower.
Defines paths, canonical label sets, thresholds, and business rule constants.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "student_data"
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
FEATURES_DATA_DIR = DATA_DIR / "features"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure runtime directories exist
for p in [DATA_DIR, CLEANED_DATA_DIR, FEATURES_DATA_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Raw files
RAW_TRAIN = RAW_DATA_DIR / "train.csv"
RAW_TEST = RAW_DATA_DIR / "test.csv"
RAW_MACRO = RAW_DATA_DIR / "macro_indicators.csv"
RAW_SKU_MASTER = RAW_DATA_DIR / "sku_master.csv"
RAW_EVENTS = RAW_DATA_DIR / "events_calendar.csv"
RAW_STOCKOUTS = RAW_DATA_DIR / "inventory_stockouts.csv"
SAMPLE_SUBMISSION = RAW_DATA_DIR / "sample_submission.csv"

# Canonical Entities
CANONICAL_REGIONS = [
    "North America",
    "Europe",
    "Asia Pacific",
    "India",
    "Latin America",
    "Middle East & Africa"
]

CANONICAL_PRODUCT_FAMILIES = [
    "Brake & Safety Systems",
    "Chassis & Structural",
    "Electrical Assemblies",
    "Hydraulic Systems",
    "Precision Machined Parts",
    "Powertrain Components",
    "Thermal Management"
]

CANONICAL_BUSINESS_UNITS = [
    "Safety & Compliance",
    "Advanced Manufacturing",
    "Electronics & Controls",
    "Fluid Control Group",
    "Power & Drive Systems"
]

FAMILY_TO_BU = {
    "Brake & Safety Systems": "Safety & Compliance",
    "Chassis & Structural": "Advanced Manufacturing",
    "Precision Machined Parts": "Advanced Manufacturing",
    "Electrical Assemblies": "Electronics & Controls",
    "Hydraulic Systems": "Fluid Control Group",
    "Thermal Management": "Fluid Control Group",
    "Powertrain Components": "Power & Drive Systems"
}

PREFIX_TO_FAMILY = {
    "BSS": "Brake & Safety Systems",
    "CSS": "Chassis & Structural",
    "ELA": "Electrical Assemblies",
    "HYD": "Hydraulic Systems",
    "PMP": "Precision Machined Parts",
    "PTC": "Powertrain Components",
    "THM": "Thermal Management"
}

CANONICAL_CHANNELS = [
    "Direct OEM",
    "Tier-1 Distributor",
    "Aftermarket",
    "Export"
]

CANONICAL_CUSTOMERS = [
    "Apex Tier One",
    "Atlas Mideast Components",
    "Crestline Manufacturing",
    "Falcon Gulf Motors",
    "Helios Auto Parts",
    "Kestrel Mobility",
    "Meridian Assembly",
    "Nexon Motor Group",
    "Orion Driveline",
    "Quantum Drive Systems",
    "Solaris Automotive",
    "Vantara Auto"
]

# Aliases and Mappings for Data Normalization
REGION_MAP = {
    "north america": "North America",
    "north america ": "North America",
    "north_america": "North America",
    "n. america": "North America",
    "na": "North America",
    "europe": "Europe",
    "europe ": "Europe",
    "eu": "Europe",
    "asia pacific": "Asia Pacific",
    "asia pacific ": "Asia Pacific",
    "asia-pacific": "Asia Pacific",
    "apac": "Asia Pacific",
    "india": "India",
    "india ": "India",
    "latin america": "Latin America",
    "latin america ": "Latin America",
    "latin-america": "Latin America",
    "latin_america": "Latin America",
    "latam": "Latin America",
    "middle east & africa": "Middle East & Africa",
    "middle east & africa ": "Middle East & Africa",
    "middle east and africa": "Middle East & Africa",
    "middle east &amp; africa": "Middle East & Africa",
    "me&a": "Middle East & Africa",
    "mea": "Middle East & Africa",
    "unassigned": "Unassigned"
}

CHANNEL_MAP = {
    "direct oem": "Direct OEM",
    "direct-oem": "Direct OEM",
    "oem direct": "Direct OEM",
    "tier-1 distributor": "Tier-1 Distributor",
    "tier 1 distributor": "Tier-1 Distributor",
    "t1 distributor": "Tier-1 Distributor",
    "aftermarket": "Aftermarket",
    "after-market": "Aftermarket",
    "export": "Export",
    "export ": "Export"
}

# Date Horizons
TRAIN_DATE_MIN = "2019-01"
TRAIN_DATE_MAX = "2026-10"
FORECAST_DATE_MIN = "2026-11"
FORECAST_DATE_MAX = "2028-05"
FORECAST_HORIZON_MONTHS = 19

# Backtest Horizons
BACKTEST_ORIGINS = ["2023-10", "2024-10", "2025-03"]
HOLDOUT_ORIGIN = "2025-03"

# Macroeconomic Columns
MACRO_COLUMNS = [
    "ISM_Manufacturing_PMI",
    "Auto_Production_YoY_Pct",
    "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index",
    "New_Vehicle_Orders_Index",
    "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index",
    "EV_Adoption_Index"
]

# Intermittency Thresholds (Syntetos-Boylan ADI-CV2 grid)
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

# Random Seed for Reproducibility
RANDOM_SEED = 42
