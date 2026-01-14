#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:08:52 2025

@author: johbay
"""
import pandas as pd

def assert_few_nans(df: pd.DataFrame, threshold: float = 0.10, columns=None):
    """
    Assert that the DataFrame contains fewer than `threshold` proportion of NaNs.

    Parameters:
    - df (pd.DataFrame): The DataFrame to check.
    - threshold (float): Maximum allowed fraction of NaNs (default is 0.10 for 10%).
    - columns (list or None): Optional list of column names to restrict the check.

    Raises:
    - AssertionError: If the fraction of NaNs exceeds the threshold.
    """
    if columns is not None:
        df = df[columns]

    total_elements = df.size
    n_nans = df.isna().sum().sum()
    nan_fraction = n_nans / total_elements

    assert nan_fraction < threshold, (
        f"Too many NaNs: {n_nans}/{total_elements} = {nan_fraction:.2%} "
        f"(Threshold: {threshold:.2%})"
    )
    

def assert_sex_column_is_binary(df: pd.DataFrame, column: str = "sex"):
    """
    Assert that all values in the 'sex' column are either 0 or 1.

    Parameters:
    - df (pd.DataFrame): DataFrame to check.
    - column (str): Name of the column to validate (default: 'sex').

    Raises:
    - AssertionError: If the column contains values other than 0 or 1.
    """
    allowed_values = {0, 1}
    actual_values = set(df[column].dropna().unique())

    assert actual_values <= allowed_values, (
        f"'{column}' column contains unexpected values: {actual_values - allowed_values}"
    )


def assert_no_negatives(df: pd.DataFrame, columns: list):
    """
    Assert that all values in the specified columns are >= 0.

    Parameters:
    - df (pd.DataFrame): DataFrame to check.
    - columns (list): List of column names to validate.

    Raises:
    - AssertionError: If any negative values are found.
    """
    for col in columns:
        if (df[col] < 0).any():
            negative_count = (df[col] < 0).sum()
            raise AssertionError(f"Column '{col}' contains {negative_count} negative value(s).")