import pandas as pd
import numpy as np

def explore():
    train = pd.read_csv('student_data/train.csv', low_memory=False)
    print("=== TRAIN DATA INSPECTION ===")
    
    # Check Units_Sold non-numeric
    non_num_units = train['Units_Sold'][pd.to_numeric(train['Units_Sold'], errors='coerce').isna()]
    print(f"Non-numeric Units_Sold count: {len(non_num_units)}")
    print(f"Sample non-numeric Units_Sold: {non_num_units.value_counts().head(10).to_dict()}")
    
    # Check Unit_Price_USD non-numeric
    non_num_price = train['Unit_Price_USD'][pd.to_numeric(train['Unit_Price_USD'], errors='coerce').isna()]
    print(f"Non-numeric Unit_Price_USD count: {len(non_num_price)}")
    print(f"Sample non-numeric Unit_Price_USD: {non_num_price.value_counts().head(10).to_dict()}")
    
    # Check Revenue_USD non-numeric
    non_num_rev = train['Revenue_USD'][pd.to_numeric(train['Revenue_USD'], errors='coerce').isna()]
    print(f"Non-numeric Revenue_USD count: {len(non_num_rev)}")
    print(f"Sample non-numeric Revenue_USD: {non_num_rev.value_counts().head(10).to_dict()}")
    
    # Check SKU_IDs
    print(f"Sample raw SKU_IDs: {train['SKU_ID'].sample(10, random_state=42).tolist()}")
    
    # Check SKU master
    sku = pd.read_csv('student_data/sku_master.csv')
    print(f"\nSKU master total SKUs: {len(sku)}")
    print(f"SKU master sample: {sku['SKU_ID'].head(5).tolist()}")
    print(f"SKU master Product Families: {sku['Product_Family'].unique()}")
    print(f"SKU master Business Units: {sku['Business_Unit'].unique()}")
    
    # Check test.csv
    test = pd.read_csv('student_data/test.csv')
    print(f"\nTest unique SKUs: {test['SKU_ID'].nunique()}")
    print(f"Test SKUs in SKU Master: {test['SKU_ID'].isin(sku['SKU_ID']).all()}")
    print(f"Test Regions: {test['Region'].unique()}")
    print(f"Test Dates: {sorted(test['Date'].unique())}")
    
    # Check stockouts
    stock = pd.read_csv('student_data/inventory_stockouts.csv')
    print(f"\nStockout SKUs: {stock['SKU_ID'].nunique()}")
    print(f"Stockout Regions: {stock['Region'].unique()}")

if __name__ == '__main__':
    explore()
