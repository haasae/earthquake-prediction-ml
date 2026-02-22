from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from .features import add_time_features


REQUIRED_COLUMNS = ["date", "time", "latitude", "longitude", "depth", "magnitude"]


@dataclass(frozen=True)
class PreprocessResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler

# Tries to map dataset columns to our expected schema.

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    colmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"lat"}:
            colmap[c] = "latitude"
        elif cl in {"lon", "long", "lng"}:
            colmap[c] = "longitude"
        elif cl in {"mag"}:
            colmap[c] = "magnitude"
        elif cl in {"dep"}:
            colmap[c] = "depth"
        elif cl in {"date"}:
            colmap[c] = "date"
        elif cl in {"time"}:
            colmap[c] = "time"
    if colmap:
        df = df.rename(columns=colmap)
    return df


def load_and_clean_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # Basic cleaning
    df = df[REQUIRED_COLUMNS].copy()
    for c in ["latitude", "longitude", "depth", "magnitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "depth", "magnitude"])
    df = df.reset_index(drop=True)

    # Add engineered time features
    df = add_time_features(df)

    # Some rows may have unparseable date/time
    for c in ["year", "month", "day", "hour", "minute"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    return df


def make_splits_and_scale(
    df: pd.DataFrame,
    target: str = "magnitude",
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: StandardScaler | None = None,
) -> PreprocessResult:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe columns.")

    # Model features
    feature_names = [
        "latitude",
        "longitude",
        "depth",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "year",
    ]
    feature_names = [f for f in feature_names if f in df.columns]

    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = scaler or StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return PreprocessResult(X_train, X_test, y_train, y_test, feature_names, scaler)


def save_processed_splits(
    df: pd.DataFrame,
    processed_dir: Path,
    test_size: float,
    random_state: int,
) -> Tuple[Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def save_scaler(scaler: StandardScaler, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path
