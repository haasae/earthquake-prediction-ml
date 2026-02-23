from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_world_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Simple scatter plot of earthquake locations."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not {"Latitude", "Longitude"}.issubset(df.columns):
        return

    plt.figure()
    plt.scatter(df["Longitude"], df["Latitude"], s=6)
    plt.title("Earthquake locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
