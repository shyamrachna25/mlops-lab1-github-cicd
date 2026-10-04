"""
This is a small set of data validation and feature engineering utilities, 
for Lab 1 (GitHub + CI/CD fundamentals) as an MLOps-flavored alternative to
the original calculator.py exercise

Each function operates on a pandas DataFrame and returns a simple, predictable
result (bool, dict, or int) so it's easy to unit test
"""

import pandas as pd
def check_missing_values(df: pd.DataFrame, threshold: float = 0.0) -> dict:
    # Checking each column of a DataFrame for NULL values.
    """
    Arguments : 
        df: The DataFrame to check
        threshold: The maximum allowed fraction of missing values per column
    Returns :
        A dictionary mapping each volumn name to True if it passes the threshold
        and False if it fails to do so
    """

    if df.empty:
        return {}
    missing_fraction = df.isnull().mean()
    return {col: bool(frac <= threshold) for col, frac in missing_fraction.items()}

def check_value_range(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> bool:
    #Checking whether value in the numeric column is within the [min_val, max_val] inclusive range
    """
    Arguments :
        df: The DataFrame to contain the column
        column: The name of the column to check
        min_val: The minimum allowed value in the range
        max_val: The maximum allowed value in the range
    Returns :
        Returns True if all non-null values in the column are within the defined range,
        otherwise False. KeyError raised if column does not exist
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    series = df[column].dropna()
    if series.empty:
        return True
    
    return bool(series.between(min_val, max_val).all())

def check_duplicate_rows(df: pd.DataFrame) -> int:
    #Counts the number of fully duplicated rows in a DataFrame.
    """
    Arguments :
        df: The DataFrame to check
    Returns :
         The number of rows that are exact duplicates of an earlier row
    """
    return int(df.duplicated().sum())

def validate_schema(df: pd.DataFrame, expected_schema: dict) -> bool:
    """
    Validates that a DataFrame's columns and dtypes match an expected schema
    exactly (same columns present, same dtypes).

    Arguments :
        df: The DataFrame to validate
        expected_schme: A dict mapping column name -> expected dtype as a 
        string, for example {"age": "int64"}
    Returns :
        True if every expected column exists with the matching dtype and no
        unexpected columns are present, False otherwise.
    """
    actual_columns = set(df.columns)
    expected_columns = set(expected_schema.keys())

    if actual_columns != expected_columns:
        return False
    
    for col, expected_dtype in expected_schema.items():
        if str(df[col].dtype) != expected_dtype:
            return False
        
    return True

def normalize_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Applies min-max normalization to a numeric column, scaling values
    to the range [0, 1].
 
    Args:
        df: The DataFrame containing the column.
        column: The name of the numeric column to normalize.
 
    Returns:
        A pandas Series of normalized values. If the column has zero
        variance (all values identical), returns a Series of zeros.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
 
    series = df[column]
    col_min = series.min()
    col_max = series.max()
 
    if col_max == col_min:
        return pd.Series([0.0] * len(series), index=series.index, name=column)
 
    return (series - col_min) / (col_max - col_min)
 