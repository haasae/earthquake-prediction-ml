from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EarthquakeEvent:
    date: str
    time: str
    latitude: float
    longitude: float
    depth: float
    magnitude: float

    @staticmethod
    def from_row(row: pd.Series) -> "EarthquakeEvent":
        return EarthquakeEvent(
            date=str(row["date"]),
            time=str(row["time"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            depth=float(row["depth"]),
            magnitude=float(row["magnitude"]),
        )


def _to_datetime_safe(date_str: str, time_str: str) -> Optional[datetime]:
    # Try common formats. Fall back to None if unparseable.
    candidates = [
        f"{date_str} {time_str}",
        f"{date_str} {time_str}:00",
    ]
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ]
    for c in candidates:
        for fmt in fmts:
            try:
                return datetime.strptime(c, fmt)
            except ValueError:
                continue
    return None

# Creates cyclical time features from date+time columns.
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dts = [
        _to_datetime_safe(str(d), str(t))
        for d, t in zip(df["date"].astype(str), df["time"].astype(str))
    ]
    dt_series = pd.to_datetime(dts, errors="coerce")

    df["year"] = dt_series.dt.year
    df["month"] = dt_series.dt.month
    df["day"] = dt_series.dt.day
    df["hour"] = dt_series.dt.hour
    df["minute"] = dt_series.dt.minute

    # cyclical encodings (handle NaNs)
    month = df["month"].fillna(1).astype(int).to_numpy()
    hour = df["hour"].fillna(0).astype(int).to_numpy()

    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return df
