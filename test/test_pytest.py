"""
test_pytest.py

Pytest-style tests for src/data_validation.py.
Run with: pytest test_pytest.py
"""

import sys
import os
import pandas as pd
import pytest

# Allow importing from the src folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_validation import (
    check_missing_values,
    check_value_range,
    check_duplicate_rows,
    validate_schema,
    normalize_column,
)


@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "age": [25, 30, 35, None, 40],
        "score": [0.1, 0.5, 0.9, 0.3, 0.7],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    })
    # Force a classic "object" dtype for the string column so this test
    # behaves the same across pandas versions (newer pandas can default
    # string columns to a native "str" dtype instead of "object").
    df["name"] = df["name"].astype(object)
    return df


@pytest.fixture
def duplicate_df():
    return pd.DataFrame({
        "a": [1, 2, 2, 3],
        "b": ["x", "y", "y", "z"],
    })


def test_check_missing_values(sample_df):
    result = check_missing_values(sample_df, threshold=0.0)
    assert result["age"] is False   # age has 1 missing value out of 5
    assert result["score"] is True  # no missing values
    assert result["name"] is True   # no missing values


def test_check_missing_values_with_threshold(sample_df):
    # Allow up to 25% missing -> age's 20% missing should now pass
    result = check_missing_values(sample_df, threshold=0.25)
    assert result["age"] is True


def test_check_value_range_pass(sample_df):
    assert check_value_range(sample_df, "score", 0.0, 1.0) is True


def test_check_value_range_fail(sample_df):
    assert check_value_range(sample_df, "score", 0.2, 0.6) is False


def test_check_value_range_missing_column(sample_df):
    with pytest.raises(KeyError):
        check_value_range(sample_df, "nonexistent", 0, 1)


def test_check_duplicate_rows_none(sample_df):
    assert check_duplicate_rows(sample_df) == 0


def test_check_duplicate_rows_found(duplicate_df):
    assert check_duplicate_rows(duplicate_df) == 1


def test_validate_schema_pass(sample_df):
    expected = {"age": "float64", "score": "float64", "name": "object"}
    assert validate_schema(sample_df, expected) is True


def test_validate_schema_fail_wrong_dtype(sample_df):
    expected = {"age": "int64", "score": "float64", "name": "object"}
    assert validate_schema(sample_df, expected) is False


def test_validate_schema_fail_missing_column(sample_df):
    expected = {"age": "float64", "score": "float64"}
    assert validate_schema(sample_df, expected) is False


def test_normalize_column(sample_df):
    normalized = normalize_column(sample_df, "score")
    assert round(normalized.min(), 5) == 0.0
    assert round(normalized.max(), 5) == 1.0


def test_normalize_column_constant_values():
    df = pd.DataFrame({"val": [5, 5, 5, 5]})
    normalized = normalize_column(df, "val")
    assert (normalized == 0.0).all()