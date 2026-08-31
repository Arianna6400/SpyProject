"""
Splits the dataset into training/validation/test sets (70/15/15) at the
session level, stratified by family: the split's unit is the session, to
avoid data leakage between sessions from the same malware campaign or
traffic profile.
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TEST_FRAC, TRAIN_FRAC, VAL_FRAC


def _split_with_fallback(df: pd.DataFrame, train_size: float, stratify_col: str, seed: int) -> tuple:
    """Stratified split by family; if a class has too few samples to be
    stratified (very small test datasets), falls back to a non-stratified
    split instead of failing outright."""
    try:
        return train_test_split(df, train_size=train_size, stratify=df[stratify_col], random_state=seed)
    except ValueError as exc:
        warnings.warn(
            f"Stratified split on '{stratify_col}' not possible ({exc}); "
            "falling back to a non-stratified split. Increase --n-per-family for a balanced split.",
            stacklevel=2,
        )
        return train_test_split(df, train_size=train_size, random_state=seed)


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
    stratify_col: str = "family",
) -> tuple:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    df_train, df_temp = _split_with_fallback(df, train_frac, stratify_col, seed)
    relative_val = val_frac / (val_frac + test_frac)
    df_val, df_test = _split_with_fallback(df_temp, relative_val, stratify_col, seed)
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )
