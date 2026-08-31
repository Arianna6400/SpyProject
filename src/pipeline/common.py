"""Utilities shared by the L1/L2 pipelines."""
from __future__ import annotations

import pandas as pd

NON_FEATURE_COLS = {"session_id", "label", "family", "y"}


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
