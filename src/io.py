from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_earthquake_csv(path: Path) -> pd.DataFrame:
    """Read earthquake CSV robustly.

    Supports both:
    - comma-separated CSV
    - whitespace-delimited (fixed-width-ish) tables that often load as a single column
    """

    df = pd.read_csv(path)
    if df.shape[1] == 1:
        # Common for files that are aligned with spaces rather than commas
        df = pd.read_csv(path, sep=r"\s+")
    return df


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )
