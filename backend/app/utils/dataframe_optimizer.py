"""
DataFrame Type Optimization Utility
Optimizes Pandas DataFrame memory usage by converting to efficient dtypes.
Critical for 4GB RAM constraint.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def optimize_dataframe_types(
    df: pd.DataFrame,
    int_columns: Optional[List[str]] = None,
    float_columns: Optional[List[str]] = None,
    category_columns: Optional[List[str]] = None,
    downcast_floats: bool = True,
    downcast_ints: bool = True,
) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting to efficient dtypes.

    Converts:
    - int64 -> int16/int32 (where possible)
    - float64 -> float32
    - object -> category (for repeated strings)

    Args:
        df: Input DataFrame
        int_columns: Optional list of integer column names to optimize
        float_columns: Optional list of float column names to optimize
        category_columns: Optional list of string columns to convert to category
        downcast_floats: Whether to downcast floats (default: True)
        downcast_ints: Whether to downcast integers (default: True)

    Returns:
        Optimized DataFrame with reduced memory footprint
    """
    if df.empty:
        return df

    df = df.copy()
    original_memory = df.memory_usage(deep=True).sum()

    # Optimize integer columns
    if downcast_ints:
        if int_columns is None:
            # Auto-detect integer columns
            int_columns = df.select_dtypes(include=["int64", "int32"]).columns.tolist()

        for col in int_columns:
            if col not in df.columns:
                continue

            col_min = df[col].min()
            col_max = df[col].max()

            # Determine optimal integer type
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype("int8")
            elif (
                col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype("int16")
            elif (
                col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype("int32")
            else:
                # Keep int64 if values are too large
                df[col] = df[col].astype("int64")

    # Optimize float columns
    if downcast_floats:
        if float_columns is None:
            # Auto-detect float columns
            float_columns = df.select_dtypes(include=["float64"]).columns.tolist()

        for col in float_columns:
            if col not in df.columns:
                continue

            # Convert float64 to float32 (saves 50% memory)
            df[col] = df[col].astype("float32")

    # Convert object columns to category (for repeated strings)
    if category_columns is None:
        # Auto-detect: convert object columns with low cardinality
        object_columns = df.select_dtypes(include=["object"]).columns.tolist()
        for col in object_columns:
            # Convert if unique values < 50% of total (e.g., team names, positions)
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                try:
                    df[col] = df[col].astype("category")
                except (ValueError, TypeError):
                    # Skip if conversion fails (e.g., mixed types)
                    continue
    else:
        for col in category_columns:
            if col not in df.columns:
                continue
            try:
                df[col] = df[col].astype("category")
            except (ValueError, TypeError):
                continue

    new_memory = df.memory_usage(deep=True).sum()
    reduction_pct = (
        ((original_memory - new_memory) / original_memory * 100)
        if original_memory > 0
        else 0
    )

    if reduction_pct > 5:  # Only log if significant reduction
        logger.debug(
            f"DataFrame memory optimized: {original_memory / 1024**2:.2f} MB -> {new_memory / 1024**2:.2f} MB ({reduction_pct:.1f}% reduction)"
        )

    return df


def optimize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick optimization: Convert all numeric columns to efficient types.

    Args:
        df: Input DataFrame

    Returns:
        Optimized DataFrame
    """
    return optimize_dataframe_types(df, downcast_floats=True, downcast_ints=True)


def optimize_categorical_columns(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert specified or auto-detected string columns to category type.

    Args:
        df: Input DataFrame
        columns: Optional list of column names to convert

    Returns:
        DataFrame with category columns
    """
    return optimize_dataframe_types(df, category_columns=columns)
