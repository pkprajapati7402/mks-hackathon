import pandas as pd
import numpy as np

def explore():
    print("=== EXPLORING RAW DATASETS ===")
    
    # 1. train.csv
    train = pd.read_csv('student_data/train.csv')
    print(f"train.csv shape: {train.shape}")
    print(f"train dates: {train['Date'].min()} to {train['Date'].max()} (unique: {train['Date'].nunique()})")
    print(f"train regions: {train['Region'].unique()}")
    print(f"train product families: {train['Product_Family'].unique()}")
    print(f"train business units: {train['Business_Unit'].unique()}")
    print(f"train sales channels: {train['Sales_Channel'].unique()}")
    print(f"train unique SKUs: {train['SKU_ID'].nunique()}")
    print(f"train null values:\n{train.isnull().sum()}")
    print(f"Units_Sold min: {train['Units_Sold'].min()}, max: {train['Units_Sold'].max()}, median: {train['Units_Sold'].median()}")
    print(f"Unit_Price_USD min: {train['Unit_Price_USD'].min()}, max: {train['Unit_Price_USD'].max()}")
    print(f"Duplicate Record_IDs: {train['Record_ID'].duplicated().sum()}")
    
    # Check date formats
    sample_dates = train['Date'].sample(min(20, len(train)), random_state=42).tolist()
    print(f"Sample train dates: {sample_dates[:5]}")
    
    # 2. test.csv
    test = pd.read_csv('student_data/test.csv')
    print(f"\ntest.csv shape: {test.shape}")
    print(f"test dates: {test['Date'].min()} to {test['Date'].max()} (unique: {test['Date'].nunique()})")
    print(f"test unique SKUs: {test['SKU_ID'].nunique()}")
    print(f"test regions: {test['Region'].unique()}")
    print(f"test nulls:\n{test.isnull().sum()}")
    
    # 3. sku_master.csv
    sku = pd.read_csv('student_data/sku_master.csv')
    print(f"\nsku_master.csv shape: {sku.shape}")
    print(f"sku_master unique SKUs: {sku['SKU_ID'].nunique()}")
    print(f"sku_master lifecycle statuses: {sku['Lifecycle_Status'].value_counts().to_dict()}")
    print(f"sku_master SKUs with Replaces_SKU: {sku['Replaces_SKU'].dropna().nunique()}")
    print(f"sku_master SKUs with EOL_Date: {sku['EOL_Date'].dropna().nunique()}")
    print(f"sku_master SKUs with Launch_Date > 2026-10: {(sku['Launch_Date'] > '2026-10').sum()}")
    
    # 4. macro_indicators.csv
    macro = pd.read_csv('student_data/macro_indicators.csv')
    print(f"\nmacro_indicators.csv shape: {macro.shape}")
    print(f"macro dates: {macro['Date'].min()} to {macro['Date'].max()}")
    print(f"macro nulls:\n{macro.isnull().sum()}")
    
    # 5. inventory_stockouts.csv
    stock = pd.read_csv('student_data/inventory_stockouts.csv')
    print(f"\ninventory_stockouts shape: {stock.shape}")
    print(f"stockout dates: {stock['Date'].min()} to {stock['Date'].max()}")
    print(f"stockout reasons: {stock['Reason'].value_counts().to_dict()}")
    print(f"stockout nulls:\n{stock.isnull().sum()}")
    
    # 6. events_calendar.csv
    events = pd.read_csv('student_data/events_calendar.csv')
    print(f"\nevents_calendar shape: {events.shape}")
    print(f"events types: {events['Event_Type'].value_counts().to_dict()}")
    print(f"events start dates: {events['Start_Date'].min()} to {events['Start_Date'].max()}")

if __name__ == '__main__':
    explore()
