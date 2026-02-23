from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .features import FeatureResult, make_features
from .io import require_columns


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


def prepare_data(
    df: pd.DataFrame,
    *,
    target: str,
    test_size: float,
    random_state: int,
) -> PreparedData:
    """Clean minimally, build features, split, scale."""

    # Accept common column names from your dataset
    # (Your file uses Date(YYYY/MM/DD) and Time)
    date_col_candidates = ["Date(YYYY/MM/DD)", "date", "Date"]
    time_col_candidates = ["Time", "time"]
    date_col = next((c for c in date_col_candidates if c in df.columns), None)
    time_col = next((c for c in time_col_candidates if c in df.columns), None)
    if date_col is None or time_col is None:
        raise ValueError(
            f"Expected date/time columns. Found columns: {list(df.columns)}"
        )

    required = [date_col, time_col, "Latitude", "Longitude", "Depth", target]
    require_columns(df, required)

    # Basic cleanup
    work = df.copy()
    work = work.dropna(subset=[target, "Latitude", "Longitude", "Depth"])

    # Ensure numeric types
    for c in ["Latitude", "Longitude", "Depth", target]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["Latitude", "Longitude", "Depth", target])

    fr: FeatureResult = make_features(
        work,
        target=target,
        date_col=date_col,
        time_col=time_col,
        numeric_cols=["Latitude", "Longitude", "Depth"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        fr.X.values,
        fr.y.values,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return PreparedData(
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test,
        feature_names=fr.feature_names,
        scaler=scaler,
    )
