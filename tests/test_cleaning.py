"""
Unit Tests for Data Cleaning Rules (Workstream 1).
"""

import pytest
import pandas as pd
import numpy as np
from src.cleaning.clean_train import parse_date_to_yyyymm, normalize_sku_id, normalize_customer_name
from src.config import CANONICAL_REGIONS, CANONICAL_CHANNELS, CANONICAL_CUSTOMERS, REGION_MAP, CHANNEL_MAP

def test_date_parsing():
    assert parse_date_to_yyyymm("2021-05") == "2021-05"
    assert parse_date_to_yyyymm("2021-05-12") == "2021-05"
    assert parse_date_to_yyyymm("2021/05") == "2021-05"
    assert parse_date_to_yyyymm("05/2021") == "2021-05"
    assert parse_date_to_yyyymm("May 2021") == "2021-05"
    assert parse_date_to_yyyymm("May-2021") == "2021-05"
    assert parse_date_to_yyyymm("2021-May") == "2021-05"

def test_sku_normalization():
    assert normalize_sku_id("ptc-001") == "PTC-001"
    assert normalize_sku_id("PTC_001") == "PTC-001"
    assert normalize_sku_id(" PTC-1 ") == "PTC-001"
    assert normalize_sku_id("PTC025") == "PTC-025"
    assert normalize_sku_id("bss-7") == "BSS-007"
    assert normalize_sku_id("HYD-045") == "HYD-045"

def test_customer_normalization():
    assert normalize_customer_name("Apex Tier One Inc.") == "Apex Tier One"
    assert normalize_customer_name("apex tier one") == "Apex Tier One"
    assert normalize_customer_name("VANTARA AUTO") == "Vantara Auto"
    assert normalize_customer_name("Atlas Mideast") == "Atlas Mideast Components"
    assert normalize_customer_name("Falcon Gulf") == "Falcon Gulf Motors"
    assert normalize_customer_name("Aex Tier One") == "Apex Tier One"
    assert normalize_customer_name("Vntara Auto") == "Vantara Auto"
    assert normalize_customer_name("Slaris Automotive") == "Solaris Automotive"
    assert normalize_customer_name("Neon Motor Group") == "Nexon Motor Group"

def test_region_mapping():
    assert REGION_MAP.get("north america") == "North America"
    assert REGION_MAP.get("eu") == "Europe"
    assert REGION_MAP.get("apac") == "Asia Pacific"
    assert REGION_MAP.get("latam") == "Latin America"
    assert REGION_MAP.get("mea") == "Middle East & Africa"
    assert REGION_MAP.get("india") == "India"
