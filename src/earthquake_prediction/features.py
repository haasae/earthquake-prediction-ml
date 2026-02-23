from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureResult:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]


def _safe_to_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """Parse separate date/time columns into a single pandas datetime.

    Tries a few common formats. Falls back to pandas' parser.
    """
    combined = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    dt = pd.to_datetime(combined, errors="coerce", utc=True)
    return dt


def add_time_features(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = _safe_to_datetime(out[date_col], out[time_col])
    out["hour"] = dt.dt.hour.fillna(0).astype(int)
    out["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
    out["month"] = dt.dt.month.fillna(1).astype(int)

    # Cyclical encoding
    out["hour_sin"] = np.sin(2 * math.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * math.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * math.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * math.pi * out["month"] / 12.0)
    return out


def make_features(
    df: pd.DataFrame,
    target: str,
    date_col: str,
    time_col: str,
    numeric_cols: list[str],
) -> FeatureResult:
    """Create a simple, defensible feature set.

    - location + depth
    - derived time features
    """
    df = add_time_features(df, date_col=date_col, time_col=time_col)
    # Keep only requested columns (and derived)
    derived = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"]
    keep = [c for c in numeric_cols if c in df.columns] + derived
    X = df[keep].copy()
    y = df[target].copy()
    return FeatureResult(X=X, y=y, feature_names=list(X.columns))
