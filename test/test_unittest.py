"""
test_unittest.py

Unittest-style tests for src/data_validation.py.
Run with: python3 -m unittest test.test_unittest
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_validation import (
    check_missing_values,
    check_value_range,
    check_duplicate_rows,
    validate_schema,
    normalize_column,
)


class TestDataValidation(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            "age": [25, 30, 35, None, 40],
            "score": [0.1, 0.5, 0.9, 0.3, 0.7],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        })
        # Force a classic "object" dtype for the string column so this
        # test behaves the same across pandas versions (newer pandas can
        # default string columns to a native "str" dtype instead).
        self.sample_df["name"] = self.sample_df["name"].astype(object)
        self.duplicate_df = pd.DataFrame({
            "a": [1, 2, 2, 3],
            "b": ["x", "y", "y", "z"],
        })

    def test_check_missing_values(self):
        result = check_missing_values(self.sample_df, threshold=0.0)
        self.assertFalse(result["age"])
        self.assertTrue(result["score"])
        self.assertTrue(result["name"])

    def test_check_missing_values_with_threshold(self):
        result = check_missing_values(self.sample_df, threshold=0.25)
        self.assertTrue(result["age"])

    def test_check_value_range_pass(self):
        self.assertTrue(check_value_range(self.sample_df, "score", 0.0, 1.0))

    def test_check_value_range_fail(self):
        self.assertFalse(check_value_range(self.sample_df, "score", 0.2, 0.6))

    def test_check_value_range_missing_column(self):
        with self.assertRaises(KeyError):
            check_value_range(self.sample_df, "nonexistent", 0, 1)

    def test_check_duplicate_rows_none(self):
        self.assertEqual(check_duplicate_rows(self.sample_df), 0)

    def test_check_duplicate_rows_found(self):
        self.assertEqual(check_duplicate_rows(self.duplicate_df), 1)

    def test_validate_schema_pass(self):
        expected = {"age": "float64", "score": "float64", "name": "object"}
        self.assertTrue(validate_schema(self.sample_df, expected))

    def test_validate_schema_fail_wrong_dtype(self):
        expected = {"age": "int64", "score": "float64", "name": "object"}
        self.assertFalse(validate_schema(self.sample_df, expected))

    def test_validate_schema_fail_missing_column(self):
        expected = {"age": "float64", "score": "float64"}
        self.assertFalse(validate_schema(self.sample_df, expected))

    def test_normalize_column(self):
        normalized = normalize_column(self.sample_df, "score")
        self.assertAlmostEqual(normalized.min(), 0.0, places=5)
        self.assertAlmostEqual(normalized.max(), 1.0, places=5)

    def test_normalize_column_constant_values(self):
        df = pd.DataFrame({"val": [5, 5, 5, 5]})
        normalized = normalize_column(df, "val")
        self.assertTrue((normalized == 0.0).all())


if __name__ == "__main__":
    unittest.main()