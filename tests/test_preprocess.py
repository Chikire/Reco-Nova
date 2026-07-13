"""Tests for the H&M data preprocessing pipeline.

Run this suite from the root of your project using: 
pytest tests/test_preprocess.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Tell Python to look in the src directory relative to this test file
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reco_nova.preprocess import (
    PipelineStats,
    _normalize_text,
    clean_customers,
    clean_items,
    clean_interactions,
    build_id_maps,
    apply_id_maps,
)


def test_normalize_text_cleans_whitespace_and_cases():
    """Verify that messy strings are flattened to lowercase with single spaces."""
    messy_series = pd.Series(["  Hello   WORLD  ", None, "", "Data   Science"])
    cleaned = _normalize_text(messy_series)
    
    assert cleaned.iloc[0] == "hello world"
    assert cleaned.iloc[1] == ""
    assert cleaned.iloc[2] == ""
    assert cleaned.iloc[3] == "data science"


def test_clean_customers_imputes_cold_start_data():
    """
    Rubric Requirement: Demographic Cold-Start.
    Verifies that missing ages are filled with the median so the UI doesn't crash 
    when making fallback recommendations for brand new users.
    """
    df = pd.DataFrame({
        "customer_id": ["user_1", "user_2", "user_3"],
        "FN": [1.0, np.nan, 1.0],
        "Active": [1.0, np.nan, 1.0],
        "club_member_status": ["ACTIVE", None, ""],
        "fashion_news_frequency": ["Regularly", "None", "None"],
        "age": [20.0, np.nan, 40.0], # Median age is 30.0
    })
    
    stats = PipelineStats()
    clean_df = clean_customers(df, stats)
    
    # Check median age imputation
    assert clean_df.loc[1, "age"] == 30.0
    
    # Check categorical fallback
    assert clean_df.loc[1, "club_member_status"] == "none"
    assert clean_df.loc[2, "club_member_status"] == "none"


def test_clean_items_builds_text_features():
    """
    Rubric Requirement: Content-Based Filtering.
    Verifies that all categorical text is combined into a single string
    so it can be safely passed to Sentence-Transformers.
    """
    df = pd.DataFrame({
        "article_id": ["item_1"],
        "prod_name": ["Red T-Shirt"],
        "product_type_name": ["T-Shirt"],
        "product_group_name": ["Garment Upper body"],
        "department_name": ["Jersey Basic"],
        "section_name": ["Mens Casual"],
        "garment_group_name": ["Jersey Casual"],
        "detail_desc": [" A plain red   shirt. "],
    })
    
    stats = PipelineStats()
    # images_dir=None allows this to run without a real folder structure
    clean_df = clean_items(df, stats, images_dir=None)
    
    # Check that the item_text is a single clean string
    expected_text = "red t-shirt t-shirt garment upper body jersey basic mens casual jersey casual a plain red shirt."
    assert clean_df.loc[0, "item_text"] == expected_text
    assert clean_df.loc[0, "has_image"] == False


def test_clean_interactions_3_way_temporal_split():
    """
    Verifies the 3-way split logic based on the max date, and ensures 
    invalid transactions (negative price, orphans) are dropped while duplicates stay.
    """
    df = pd.DataFrame({
        "customer_id": ["u1", "u1", "u2", "u3", "u4", "u5", "u6"],
        "article_id": ["i1", "i1", "i2", "i3", "i4", "i5", "i1"],
        "price": [10.0, 10.0, 15.0, 20.0, -5.0, 12.0, 10.0], 
        "sales_channel_id": [2, 2, 1, 2, 1, 2, 1],
        "t_dat": [
            "2020-09-22", # 1. Valid Test set (max date, Week 4 of month 6)
            "2020-09-22", # 2. DUPLICATE PURCHASE (should be kept)
            "2020-09-17", # 3. Valid Validation set (Week 3 of month 6)
            "2020-07-01", # 4. Valid Train set (Month 4)
            "2020-07-01", # 5. Dropped: Negative price
            "2020-07-02", # 6. Dropped: Orphan item ('i5' is missing from items)
            "2019-01-01", # 7. Dropped: Outside 6-month window
        ]
    })
    
    # Notice 'i5' is intentionally missing from this valid set
    valid_articles = {"i1", "i2", "i3", "i4"}
    stats = PipelineStats()
    
    train_df, val_df, test_df = clean_interactions(df, valid_articles, stats)
    
    # Verifying sizes (only valid rows should survive)
    assert len(train_df) == 1, "Only u3 should be in train"
    assert len(val_df) == 1, "Only u2 should be in validation"
    assert len(test_df) == 2, "u1 (and their duplicate purchase) should be in test"
    
    # Verifying the duplicate was kept for quantity signals
    assert test_df.iloc[0]["article_id"] == "i1"
    assert test_df.iloc[1]["article_id"] == "i1"
    
    # Verifying stats counters actually caught the specific bad rows
    assert stats.interactions_dropped_invalid_price == 1  # Caught u4
    assert stats.interactions_dropped_orphan_items == 1   # Caught u5


def test_id_mapping_prevents_data_leakage():
    """
    Crucial Test: Ensures new users/items in the validation or test sets 
    are mapped to NaN rather than being assigned an ID, simulating a true cold-start.
    """
    train_df = pd.DataFrame({
        "customer_id": ["user_a", "user_b"],
        "article_id": ["item_a", "item_b"]
    })
    
    test_df = pd.DataFrame({
        "customer_id": ["user_a", "user_c"], # user_c is brand new
        "article_id": ["item_b", "item_c"]   # item_c is brand new
    })
    
    # Build maps ONLY on train
    user_map, item_map = build_id_maps(train_df)
    
    # Apply maps to both
    train_mapped = apply_id_maps(train_df, user_map, item_map)
    test_mapped = apply_id_maps(test_df, user_map, item_map)
    
    # Train mapping should be flawless
    assert not train_mapped["user_idx"].isna().any()
    assert not train_mapped["item_idx"].isna().any()
    
    # Test mapping must recognize cold starts (user_c and item_c should be NaN)
    assert pd.notna(test_mapped.loc[0, "user_idx"]) # user_a is known
    assert pd.isna(test_mapped.loc[1, "user_idx"])  # user_c is unknown
    
    assert pd.notna(test_mapped.loc[0, "item_idx"]) # item_b is known
    assert pd.isna(test_mapped.loc[1, "item_idx"])  # item_c is unknown